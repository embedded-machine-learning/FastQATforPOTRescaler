#import argparse
#import io
#import json

from email.policy import strict
from torch.utils.data import DataLoader
import torch.nn as nn

#from torchvision.utils import save_image

from utils.datasets import *
from utils.utils import *

from model.model import *

from PIL import Image
import time
import xml.dom.minidom
import pathlib
import cv2
import sys

img_name = ""
SAVE_OUT = False                #   bs => 1 , save all intermediate values
                                #       Zeile 303 ändern
                                #       Tom verifiziert damit
SAVE_HARDWARE = False           #   saves npz file fpr fpga
SAVE_ZIP = False                #   for copying
SAVE_XML = False                #   saves for mathias 

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

class QTCamtadNetFixed(nn.Module):
    def __init__(self):
        super(QTCamtadNetFixed, self).__init__()

        self.n = []
        self.t = []
        self.num_bits = 8
        self.out_bits = 20
        self.exp = 0
        self.run = 0

        self.layers = nn.Sequential(
            
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=int(np.floor(3/2)), groups=1, bias = False),
            nn.LeakyReLU(0.125),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=int(np.floor(3/2)), groups=1, bias = False),
            nn.LeakyReLU(0.125),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=int(np.floor(3/2)), groups=1, bias = False),
            nn.LeakyReLU(0.125),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=int(np.floor(3/2)), groups=1, bias = False),
            nn.LeakyReLU(0.125),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=int(np.floor(3/2)), groups=1, bias = False),
            nn.LeakyReLU(0.125),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=int(np.floor(3/2)), groups=1, bias = False),
            nn.LeakyReLU(0.125),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=int(np.floor(3/2)), groups=1, bias = False),
            nn.LeakyReLU(0.125),

            nn.Conv2d(64, 36, kernel_size=1, stride=1, padding=int(np.floor(1/2)), groups=1, bias = False),
        )

        self.yololayer = YOLOLayer(
            [[20, 20], [20, 20], [20, 20], [20, 20], [20, 20], [20, 20]])
        
        self.yolo_layers = [self.yololayer]

    def forward(self, x):
        if SAVE_OUT:
            global img_name
            img_name =  img_name.split("/")[-1].split(".")[0]
            print(img_name)
        img_size = x.shape[-2:]
        # print(x.shape)
        yolo_out, out = [], []

        min_val = torch.tensor(-(1 << (self.num_bits - 1)))
        max_val = torch.tensor((1 << (self.num_bits - 1))-1)

        x = x-0.5
        x = x.clamp(-0.5,0.5)

        delta = 1.0/(2.0**(-self.run)-1)

        x = x/delta
        x = torch.floor(x)

        # x = x-127
        if torch.max(x.view(-1)) > (2**7-1) or torch.min(x.view(-1))<-2**7:
            print("input out of bounds")

        x = torch.clamp(x,min_val,max_val)

        if SAVE_OUT:
            with open("out/" + img_name + "input.npz", 'wb') as f:
                print(x.detach().cpu().numpy().astype(np.int32).shape)
                x.detach().cpu().numpy().astype(np.int32).tofile(f)
                # print(self.run)
                # print(np.max(x.detach().cpu().numpy().astype(np.int32)))
                # print(np.min(x.detach().cpu().numpy().astype(np.int32)))
        
        if SAVE_OUT:
            j = 0
        for i, layer in enumerate(self.layers):
            
            if "Conv2d" in str(layer):
                if SAVE_OUT:
                    with open("out/" + img_name + "input_layer" + str(j) + ".npz", 'wb') as f:
                        print("input_layer" + str(j))
                        print(x.detach().cpu().numpy().astype(np.int32).shape)
                        # x.detach().cpu().numpy().astype(np.int8).tofile(f)
                        # temp = np.ascontiguousarray(x.moveaxis(1,3).detach().cpu().numpy().astype(np.int8))
                        temp = np.ascontiguousarray(x.detach().cpu().numpy().astype(np.int32))
                        temp.tofile(f)
                    with open("out/" + img_name + "used_weights_layer" + str(j) + ".npz", 'wb') as f:
                        temp = np.ascontiguousarray(layer.weight.data.detach().cpu().numpy().astype(np.int8))
                        temp.tofile(f)
                
                x = layer(x)

                if SAVE_OUT:
                    with open("out/" + img_name + "output_conv_layer" + str(j) + ".npz", 'wb') as f:
                        temp = np.ascontiguousarray(x.detach().cpu().numpy().astype(np.int32))
                        temp.tofile(f)

                x = torch.floor(x)
                
                x = torch.clamp(x,torch.tensor(-(1 << (self.out_bits - 1))),torch.tensor((1 << (self.out_bits - 1))-1))
                if torch.max(torch.abs(x)) > 2**(self.out_bits-1):
                    print("conv2d" + str(torch.max(torch.abs(x))))
                x = x*torch.exp2(self.n[i])[None, :, None, None] + self.t[i][None, :, None, None]
               
                x = torch.floor(x)
                # print("conv2d scale" + str(torch.max(torch.abs(x))))
                if i != len(self.layers)-1:
                    x = torch.clamp(x,min_val,max_val)
                    # print(j)
                else:#
                    # print(j)
                    x = torch.clamp(x,(-2**15),(2**15)-1)
                    # print(torch.max(x))
                    # print(torch.min(x))

                if SAVE_OUT:
                    with open("out/" + img_name + "n_layer" + str(j) + ".npz", 'wb') as f:
                        temp = np.ascontiguousarray(self.n[i].detach().cpu().numpy().astype(np.int8))
                        temp.tofile(f)

                    with open("out/" + img_name + "t_layer" + str(j) + ".npz", 'wb') as f:
                        temp = np.ascontiguousarray(self.t[i].detach().cpu().numpy().astype(np.int8))
                        temp.tofile(f)
                    with open("out/" + img_name + "output_layer" + str(j) + ".npz", 'wb') as f:
                        print("output_layer" + str(j))
                        print(x.detach().cpu().numpy().astype(np.int32).shape)
                        # x.detach().cpu().numpy().astype(np.int32).tofile(f)
                        # temp = np.ascontiguousarray(x.moveaxis(1,3).detach().cpu().numpy().astype(np.int32))
                        temp = np.ascontiguousarray(x.detach().cpu().numpy().astype(np.int32))
                        temp.tofile(f)
                    
                # print(x.shape)
                # print(np.ascontiguousarray(x.detach().cpu().numpy().astype(np.int32))[0,0,0])
                # input()
            elif "LeakyReLU" in str(layer):
            # elif "ReLU" in str(layer):
                x = layer(x)
                x = torch.floor(x)
                # print("relu" + str(torch.max(torch.abs(x)))
                x = torch.clamp(x,min_val,max_val)
                if SAVE_OUT:
                    with open("out/" + img_name + "output_relu" + str(j) + ".npz", 'wb') as f:
                        print("output_relu" + str(j))
                        print(x.detach().cpu().numpy().astype(np.int32).shape)
                        # temp = np.ascontiguousarray(x.moveaxis(1,3).detach().cpu().numpy().astype(np.int8))
                        temp = np.ascontiguousarray(x.detach().cpu().numpy().astype(np.int32))
                        temp.tofile(f)
                        j = j+1
                        #x.detach().cpu().numpy().astype(np.int32).tofile(f)
            elif "MaxPool2d" in str(layer):
                x = layer(x)
            else:
                print("!!!!!!!!!!!!!!!!!!!!!!!!!")
                print(layer)
            

        x = x/(2**-self.exp[None,:,None,None])
        # print(self.exp)
        x = self.yololayer(x, img_size)

        yolo_out.append(x)

        if self.training:  # train
            return yolo_out
        else:  # test
            io, p = zip(*yolo_out)  # inference output, training output
            return torch.cat(io, 1), p
        return x

def save_results_xml(results):
    doc = xml.dom.minidom.Document()
    root = doc.createElement("results")
    
    for i, result in enumerate(results):
        rectangle = result[0]
        image_e = root.appendChild(doc.createElement("image"))

        doc.appendChild(root)
        name_e = doc.createElement("filename")
        name_t = doc.createTextNode(result[1])
        name_e.appendChild(name_t)
        image_e.appendChild(name_e)

        size_e = doc.createElement("size")
        node_width = doc.createElement("width")
        node_width.appendChild(doc.createTextNode("640"))
        node_length = doc.createElement("length")
        node_length.appendChild(doc.createTextNode("360"))
        size_e.appendChild(node_width)
        size_e.appendChild(node_length)
        image_e.appendChild(size_e)

        object_node = doc.createElement("object")
        node_bnd_box = doc.createElement("bndbox")
        node_bnd_box_xmin = doc.createElement("xmin")
        node_bnd_box_xmin.appendChild(doc.createTextNode(str(rectangle[0].detach().cpu().numpy())))
        node_bnd_box_xmax = doc.createElement("xmax")
        node_bnd_box_xmax.appendChild(doc.createTextNode(str(rectangle[1].detach().cpu().numpy())))
        node_bnd_box_ymin = doc.createElement("ymin")
        node_bnd_box_ymin.appendChild(doc.createTextNode(str(rectangle[2].detach().cpu().numpy())))
        node_bnd_box_ymax = doc.createElement("ymax")
        node_bnd_box_ymax.appendChild(doc.createTextNode(str(rectangle[3].detach().cpu().numpy())))
        node_bnd_box.appendChild(node_bnd_box_xmin)
        node_bnd_box.appendChild(node_bnd_box_xmax)
        node_bnd_box.appendChild(node_bnd_box_ymin)
        node_bnd_box.appendChild(node_bnd_box_ymax)

        object_node.appendChild(node_bnd_box)
        image_e.appendChild(object_node)

    file_name =  "out/results.xml"
    with open(file_name, "w") as fp:
        doc.writexml(fp, indent="\t", addindent="\t", newl="\n", encoding="utf-8")

if __name__ == '__main__':

    #Weights from training
    weights = "weights/qat_network.pt"

    #Adjustment File
    adjustmentfile = "out/adjustment.txt"

    #Outfile for hardware repo
    outfile = "out/camtad.npz"

    #Dir for test dataa
    # path = '/home/dschnoell/sources/sim2/data_training'
    path = '/home/dschnoell/sources/sim2/test_images'
    
    print_images = True

    batch_size = 256
    if SAVE_XML or SAVE_OUT:
        if batch_size != 1:
            batch_size = 1

    img_size = 320

    #Use cuda if avaliable
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # torch.cuda.set_device(1)
    #Camtad 8bit unsigned int fake quantized network
    model = CamtadNetFixedPoolN2().to(device)

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
            elif "BlockQuantN" in str(layer):
                # print(i)
                qtmodel.t.append(torch.round(layer.bn.t))
                # print(layer.bn.t)
                qtmodel.n.append(torch.round(layer.bn.n))
                # print(torch.round(layer.bn.inference_n)[:])
                #print(layer.bn.t[0])
                #print(layer.bn.inference_n[0])
                qtmodel.layers[i].weight.data = torch.round(layer.conv.used_weights.data)
                
                #Activation
                qtmodel.t.append(None)
                qtmodel.n.append(None)
                i = i + 2 # conv + activation
            elif "MaxPool" in str(layer):
                qtmodel.t.append(None)
                qtmodel.n.append(None)
                i = i + 1
            # elif "Conv2d" in str(layer):
            #     qtmodel.layers[i].weight.data = torch.round(layer.used_weights.data)
            #     i = i+1
            elif "Stop" in str(layer):
                qtmodel.exp =layer.exp
            else:
                print("!!!!!!!!!!!!!")
                print(layer)
                #print(layer.exponent[0,0,0])
        if SAVE_HARDWARE:
            shutil.copyfile(weights, "out/qat_network.pt")
            #Save data to file for Hardware Team
            f.write(str(int(qtmodel.run.detach().cpu()))+ "\n")
            f.write(str(int(qtmodel.exp[:].detach().cpu()))+ "\n")
            j = 0
            for i, layer in enumerate(qtmodel.layers):
                if "Conv2d" in str(layer):
                    #Add weights and bias to numpy dictionary
                    layer_dict["w_conv"+str(j)] = layer.weight.data.detach().cpu().numpy().astype(np.int8)
                    
                    print(qtmodel.t[i].detach().cpu().numpy().astype(np.int16))
                    print(qtmodel.n[i][:].detach().cpu().numpy().astype(np.int8))
                    layer_dict["b_conv"+str(j)] = qtmodel.t[i].detach().cpu().numpy().astype(np.int16)
                    layer_dict["a_conv"+str(j)] = qtmodel.n[i][:].detach().cpu().numpy().astype(np.int8)
                        # print(layer_dict["a_conv"+str(j)])
                    j = j + 1
                elif "LeakyReLU":
                    pass
                elif "MaxPool2d":
                    pass
                else:
                    print("!!!!!!!!!!")
                    print(layer) 

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
    results = []

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
            img_name = paths[0].split("/")[-1]
            inf_out, train_out = model(imgs)  # inference and training outputs
            # print(inf_out.shape)
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
            if SAVE_XML:
                b1_x1 = torch.round(b1_x1)
                b1_x2 = torch.round(b1_x2)
                b1_y1 = torch.round(b1_y1)
                b1_y2 = torch.round(b1_y2)

            if SAVE_XML:
                results.append([[b1_x1,b1_x2,b1_y1,b1_y2], img_name])

            # pre_box = get_boxes(inf_out[..., :4], inf_out[..., 4])
            tbox = targets[..., 2:6] * torch.Tensor([width, height, width, height]).to(device)

            ious = bbox_iou(pre_box, tbox)
            iou_sum += ious.sum()
            loss_o = loss / (batch_i + 1)

            iou = iou_sum / test_n
            s = ('%10.4f')*4 % (iou, loss_o.sum(), loss_o[0], loss_o[1])
            pbar.set_description(s)
    # print(results)
    if SAVE_XML:
        # paths = [dataloader.dataset[i][2] for i, _ in enumerate(dataloader.dataset)]
        save_results_xml(results)
    if SAVE_ZIP:
        shutil.make_archive("out", 'zip', "out/")
