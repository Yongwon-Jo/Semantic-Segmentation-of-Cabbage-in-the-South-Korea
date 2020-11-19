class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'cabbage':
            return './mypath/cabbage'

        # If you want a path to another dataset, add it as elif like 5-channel dataset
        elif dataset == 'cabbage5channel':
            return './mypath/cabbage5channel'

        else:
            print(f'Dataset {dataset} is not available.')
            raise NotImplementedError
