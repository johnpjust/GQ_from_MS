import statsmodels.api as sm
import numpy as np
from GQ_MS import GQ_MS

def load_dataset(args):

    np.random.seed(args.manualSeed)
    if args.dataset == 'gq_ms_wheat':
        dataset = GQ_MS('GQ_MS/wheat_perms.xlsx', normalize=args.normalize, logxfm = args.xfm)
    elif args.dataset == 'gq_ms_soy':
        dataset = GQ_MS('GQ_MS/soy_perms.xlsx', normalize=args.normalize, logxfm = args.xfm)
    elif args.dataset == 'gq_ms_corn':
        dataset = GQ_MS('GQ_MS/corn_perms.xlsx', normalize=args.normalize, logxfm = args.xfm)
    elif args.dataset == 'gq_ms_canola':
        dataset = GQ_MS('GQ_MS/canola_perms.xlsx', normalize=args.normalize, logxfm = args.xfm)
    elif args.dataset == 'gq_ms_barley':
        dataset = GQ_MS('GQ_MS/barley_perms.xlsx', normalize=args.normalize, logxfm = args.xfm)
    else:
        raise RuntimeError()

    dataset_train = dataset.trn.x.astype(np.float64)
    dataset_valid = dataset.val.x.astype(np.float64)
    dataset_test = dataset.tst.x.astype(np.float64)

    return dataset_train, dataset_valid, dataset_test


class parser_:
    pass

def main():
    args = parser_()
    args.dataset = 'gq_ms_wheat'
    args.manualSeed = 1
    args.normalize = True
    args.xfm = True
    data_loader_train, data_loader_valid, data_loader_test = load_dataset(args)

    dens_u = sm.nonparametric.KDEMultivariate(data=data_loader_train, var_type='cccc')

if __name__ == '__main__':
    main()

