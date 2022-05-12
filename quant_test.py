#import argparse
#import io
#import json

from torch.utils.data import DataLoader
#from torchvision.utils import save_image

from utils.datasets import *
from utils.utils import *

from model_server.model import *

from PIL import Image

def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes
    """
 
    # Transform from center and width to exact coordinates
    b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
    b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
    b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
    b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    
    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def get_boxes(pred_boxes, pred_conf):
    n = pred_boxes.size(0)
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor
    p_boxes = FloatTensor(n, 4)

    for i in range(n):
        _, index = pred_conf[i].max(0)
        p_boxes[i] = pred_boxes[i][index]

    return p_boxes

           
if __name__ == '__main__':

    #Weights from training
    weights = "weights/test_best.pt"

    #Adjustment File
    adjustmentfile = "out/adjustment.txt"

    #Outfile for hardware repo
    outfile = "out/camtad.npz"

    #Dir for test dataa
    path = '//binfl/lv71513/ddallinger/datasets/dac_contest_last/data_test'

    batch_size = 32
    img_size = 320

    #Use cuda if avaliable
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Camtad 8bit unsigned int fake quantized network
    model = CamtadNetFixed().to(device)
    model.setquant(0)

    # Load weights
    model.load_state_dict(torch.load(weights, map_location=device)['model'])

    #Readl Quantized 8bit unsigned int model
    qtmodel = QTCamtadNetFixed().to(device)

    layer_dict = {}

    i = 0
    with open(adjustmentfile, "w") as f:
        for layer in model.layers:
            if "Start" in str(layer):
                qtmodel.run = layer.run
                #print(layer.run)
            if "BlockQuant" in str(layer):
                qtmodel.t.append(layer.bn.t)
                qtmodel.n.append(layer.bn.inference_n)
                #print(layer.bn.t[0])
                #print(layer.bn.inference_n[0])
                qtmodel.layers[i].weight.data = layer.conv.used_weights.data
                #RELU
                qtmodel.t.append(None)
                qtmodel.n.append(None)
                i = i+2
            elif "Conv2dQuant" in str(layer):
                qtmodel.layers[i].weight.data = layer.used_weights.data
                i = i+1
            elif "Stop" in str(layer):
                qtmodel.exponent = layer.exponent
                #print(layer.exponent[0,0,0])
                i = i+1
            #print()
    # exit()
    #             #Add weights and bias to numpy dictionary
    #             layer_dict["w_conv"+str(j)] = qtmodel.layers[i].weight.data.detach().cpu()
    #             layer_dict["b_conv"+str(j)] = qtmodel.layers[i].bias.data.detach().cpu()

    #             if i != len(model.layers):
    #                 if ("Conv2d") in str(qtmodel.layers[i+1]):
    #                     activation_scale = qtmodel.input_quantizer[i+1].scale
    #                 if ("Conv2d") in str(qtmodel.layers[i+2]):
    #                     activation_scale = qtmodel.input_quantizer[i+2].scale

    #                 adjustment = ((qtmodel.weight_quantizer[i].scale * qtmodel.input_quantizer[i].scale) / activation_scale)
    #                 layer_dict["a_conv"+str(j)] = abs(round(math.log2(adjustment)))
                    
    #             j+=1

    #             #If first layer add the input quantization value
    #             if i == 0:
    #                 # f.write(str(i) + " input: " + str(round(math.log2(qtmodel.input_quantizer[0].scale)))+ "\n")
    #                 f.write(str(abs(round(math.log2(qtmodel.input_quantizer[0].scale))))+ "\n")
                
    #             # f.write(str(i) + " adjustment: " + str(round(math.log2(adjustment)))+ "\n")
    #             f.write(str(abs(round(math.log2(adjustment))))+ "\n")
                
    #             #If last layer add output dequantization value
    #             if i == 11:
    #                 # f.write(str(i) + " output: " + str(round(math.log2(qtmodel.activation_quantizer[12].scale)))+ "\n")
    #                 f.write(str(abs(round(math.log2(qtmodel.activation_quantizer[11].scale))))+ "\n")

    #Save quantized network
    torch.save(qtmodel.state_dict(), "out/quantized_net.pt")

    #Save numpy dictionary for hardware repo
    np.savez(outfile, **layer_dict)

    #-------------- Quantized Model Simultion --------------#
    # model = qtmodel
    model = qtmodel
    model.nc = 1

    # Dataloader
    dataset = LoadImagesAndLabels(path, img_size, batch_size, rect=False, cache_labels=False)
    batch_size = min(batch_size, len(dataset))
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8]),
                            pin_memory=True,
                                collate_fn=dataset.collate_fn)

    model.eval()
    loss = torch.zeros(3)
    iou_sum = 0
    test_n = 0

    print(('\n' + '%10s' * 4) % ('IOU', 'l', 'Giou-l', 'obj-l'))
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))    
    for batch_i, (imgs, targets, paths, shapes) in pbar:

        imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
        if batch_i == 0:
            min_val = torch.tensor(-(1 << (8 - 1)))
            max_val = torch.tensor((1 << (8 - 1))-1)

        targets = targets.to(device)
        bn, _, height, width = imgs.shape  # batch size, channels, height, width
        test_n += bn

        # Disable gradients
        with torch.no_grad():
            # Run model
            inf_out, train_out = model(imgs)  # inference and training outputs
            
            # Compute loss
            if hasattr(model, 'hyp'):  # if model has loss hyperparameters
                loss += compute_loss(train_out, targets, model)[1][:3].cpu()  # GIoU, obj, cls

            inf_out = inf_out.view(inf_out.shape[0], 6, -1)
            # ft = torch.cuda.FloatTensor if p[0].is_cuda else torch.Tensor
            inf_out_t = torch.zeros_like(inf_out[:, 0, :])
            for i in range(inf_out.shape[1]):
               inf_out_t += inf_out[:, i, :]
            inf_out_t = inf_out_t.view(inf_out_t.shape[0], -1, 6) / 6


            pre_box = get_boxes(inf_out_t[..., :4], inf_out_t[..., 4])

            box1 = pre_box[0] / torch.Tensor([width, height, width, height]).to(device) * torch.Tensor([640,360,640,360]).to(device)
            b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
            b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
            #print bounding box
            #print(str(int(b1_x1)) + " " + str(int(b1_y1)) + " " + str(int(b1_x2))  + " " + str(int(b1_y2)) )
            #exit()

            # pre_box = get_boxes(inf_out[..., :4], inf_out[..., 4])
            tbox = targets[..., 2:6] * torch.Tensor([width, height, width, height]).to(device)

            ious = bbox_iou(pre_box, tbox)
            iou_sum += ious.sum()
            loss_o = loss / (batch_i + 1)

            iou = iou_sum / test_n
            s = ('%10.4f')*4 % (iou, loss_o.sum(), loss_o[0], loss_o[1])
            pbar.set_description(s)