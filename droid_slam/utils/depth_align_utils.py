import torch

def align_scale_and_shift(prediction, target, weights):

    '''
    weighted least squares problem to solve scale and shift: 
        min sum{ 
                  weight[i,j] * 
                  (prediction[i,j] * scale + shift - target[i,j])^2 
               }

    prediction: [B,H,W]
    target: [B,H,W]
    weights: [B,H,W]
    '''

    if weights is None:
        weights = torch.ones_like(prediction).to(prediction.device)
    if len(prediction.shape)<3:
        prediction = prediction.unsqueeze(0)
        target = target.unsqueeze(0)
        weights = weights.unsqueeze(0)  
    a_00 = torch.sum(weights * prediction * prediction, dim=[1,2])
    a_01 = torch.sum(weights * prediction, dim=[1,2])
    a_11 = torch.sum(weights, dim=[1,2])
    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(weights * prediction * target, dim=[1,2])
    b_1 = torch.sum(weights * target, dim=[1,2])
    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b            
    det = a_00 * a_11 - a_01 * a_01
    scale = (a_11 * b_0 - a_01 * b_1) / det
    shift = (-a_01 * b_0 + a_00 * b_1) / det
    error = (scale[:,None,None]*prediction+shift[:,None,None]-target).abs()
    masked_error = error*weights
    error_sum = masked_error.sum(dim=[1,2])
    error_num = weights.sum(dim=[1,2])
    avg_error = error_sum/error_num

    return scale,shift,avg_error
