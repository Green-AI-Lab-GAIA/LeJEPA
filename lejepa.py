import os, torch, time, yaml
from torchvision.datasets import CIFAR10

from torchvision.models import resnet18, resnet50, ResNet50_Weights, ResNet18_Weights

from utils import AverageMeter, CSVLogger
from transforms import MultiviewTransform, common_transform
from evaluate import evaluate

# http://arxiv.org/abs/2511.08544
# SigReg minimizes Epps-Pulley statistic in M random projections
def sigreg(x, M = 256):
    N, C = x.size() # N is batch size, C is embedding dim
    
    A = torch.randn(C, M, device=x.device)
    A = torch.nn.functional.normalize(A, dim=0)

    t = torch.linspace(-5., 5., 17, device=x.device)
    exp_f = torch.exp(-0.5 * t ** 2)

    x_t = (x @ A).unsqueeze(2) * t #(N, M, T)

    mean_cos = torch.cos(x_t).mean(0) # (M, T)
    mean_sin = torch.sin(x_t).mean(0) # (M, T)
    err = ((mean_cos - exp_f).square() + mean_sin.square()).mul(exp_f)

    T = torch.trapezoid(err, t, dim=1) # (M,)

    return T.mean() * N

# Initialize the model and modify for CIFAR-10
def init_model(params):
    
    if params['model'] == 'resnet18':
        encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        encoder.fc = torch.nn.Linear(512, params['embedding_dim'])
    elif params['model'] == 'resnet50':
        encoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        encoder.fc = torch.nn.Linear(2048, params['embedding_dim'])

    # http://arxiv.org/abs/2002.05709
    # Because CIFAR-10 images are much smaller than ImageNet images,
    # we replace the first 7x7 Conv of stride 2 with 3x3 Conv of stride 1, 
    # and also remove the first max pooling operation.
    encoder.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    encoder.maxpool = torch.nn.Identity()

    return encoder

# Initialize the optimizer excluding weight decay for biases and normalization parameters
def init_opt(encoder, params):

    param_groups = [
        {
            'params': [p for p in encoder.parameters() if len(p.squeeze().shape) > 1],
            'weight_decay': float(params['weight_decay'])
        },
        {
            'params': [p for p in encoder.parameters() if len(p.squeeze().shape) <= 1],
            'weight_decay': 0
        }
    ]
    
    return torch.optim.AdamW(param_groups, lr=float(params['lr']))


def save_checkpoint(path, params, epoch, encoder, optimizer):
    save_path = os.path.join(params['folder'], params['name'], path)

    save_dict = {
        'encoder': encoder.state_dict(),
        'opt': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(save_dict, save_path)


def main():

    with open('config.yaml', 'r') as f:
        params = yaml.safe_load(f)

    params['name'] = input("Enter a name for this run: ")
    
    os.makedirs(os.path.join(params['folder'], params['name']), exist_ok=True)
    os.system(f'cp config.yaml {os.path.join(params["folder"], params["name"], "config.yaml")}')
    
    logger = CSVLogger(os.path.join(params['folder'], params['name'], 'log.csv'),
                          'epoch', 'pred_loss', 'sigreg_loss', 'accuracy', 'epoch_time')
    
    transform = MultiviewTransform(nglobal=params['global_views'], nlocal=params['local_views'])
    dataset = CIFAR10(root='./data', train=True, transform=common_transform, download=True)

    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=params['batch_size'],
        num_workers=params['num_workers'],
        shuffle=True,
        drop_last=True)

    encoder = init_model(params).to(params['device'])
    encoder.train()

    optimizer = init_opt(encoder, params)
    best_accuracy = 0

    # -- TRAINING LOOP
    for epoch in range(params['epochs']):
        epoch_start_time = time.time()
        pred_meter = AverageMeter()
        sigreg_meter = AverageMeter()

        for data in dataloader:
            optimizer.zero_grad()

            # Step 0. load data to gpu
            views, labels = data
            views = transform(views.to(params['device']))

            # Step 1. forward pass
            all_views = torch.cat(views, dim=0) # [(global_views + local_views) * batch_size, C, H, W]
            all_emb = encoder(all_views)
            all_emb = all_emb.view(params['global_views'] + params['local_views'], params['batch_size'], -1)
            emb_g = all_emb[:params['global_views']]
            emb_l = all_emb[params['global_views']:]

            # Step 2. compute sigreg loss
            sigreg_loss = torch.stack([sigreg(o) for o in emb_l]).mean()

            # Step 3. compute prediction loss
            center = emb_g.mean(0, keepdim=True)
            pred_loss = (emb_l - center).square().mean()
            
            lambd = params['lambda']
            loss = (1 - lambd) * pred_loss + lambd * sigreg_loss

            # Step 4. Optimization step
            loss.backward()
            optimizer.step()
        
            # Step 5. logging
            pred_meter.update(pred_loss.item())
            sigreg_meter.update(sigreg_loss.item())
            print(f"Pred Loss: {pred_loss.item():.4f}, SigReg Loss: {sigreg_loss.item():.4f}, Total Loss: {loss.item():.4f}")

        accuracy = evaluate(encoder, params['device'])
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save_checkpoint('best_epoch.pth.tar', params, epoch + 1, encoder, optimizer)

        # -- Save Checkpoint after every epoch
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch}: Pred Loss: {pred_meter.avg:.4f}, SigReg Loss: {sigreg_meter.avg:.4f}. Epoch Time: {epoch_time:.2f}s")
        save_checkpoint('latest_epoch.pth.tar', params, epoch + 1, encoder, optimizer)
        logger.log({
            'epoch': epoch,
            'pred_loss': pred_meter.avg,
            'sigreg_loss': sigreg_meter.avg,
            'accuracy': accuracy,
            'epoch_time': epoch_time,
        })


if __name__ == '__main__':
    main()