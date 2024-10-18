from model.gslsanet import DCRNet


def get_model(opt, convnet):
    if opt.model_name == 'DCRNet':
        model = DCRNet(opt, convnet, inplanes=opt.inplanes)
    else:
        raise ValueError('Unknown Model')
    return model
