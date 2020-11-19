from dataloaders.dataset import Cabbage, Cabbage_5channel
from torch.utils.data import DataLoader


def make_data_loader(args, **kwargs):

    if args.dataset == 'cabbage':
        train_set = Cabbage.CabbageDataset(args, split='train')
        val_set = Cabbage.CabbageDataset(args, split='val')

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, num_class

    elif args.dataset == 'cabbage5channel':
        train_set = Cabbage_5channel.Cabbage5ChannelDataset(args, split='train')
        val_set = Cabbage_5channel.Cabbage5ChannelDataset(args, split='val1')

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, num_class

    else:
        raise NotImplementedError
