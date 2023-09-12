import torchvision
import torch
import torch.nn as nn
import torchvision.transforms.functional as functional

to_compare = {
    "bunny_format_f16.png": "bunny_format_f32.png",
    "bunny_format_unorm.png": "bunny_format_f32.png",
    "bunny_format_bc7.png": "bunny_format_f32.png",

    "disney_cloud_format_f16.png": "disney_cloud_format_f32.png",
    "disney_cloud_format_unorm.png": "disney_cloud_format_f32.png",
    "disney_cloud_format_bc7.png": "disney_cloud_format_f32.png",

    "dragon_format_f16.png": "dragon_format_f32.png",
    "dragon_format_unorm.png": "dragon_format_f32.png",
    "dragon_format_bc7.png": "dragon_format_f32.png",
    
    "fire_format_f16.png": "fire_format_f32.png",
    "fire_format_unorm.png": "fire_format_f32.png",
    "fire_format_bc7.png": "fire_format_f32.png",
}



for comp, base in to_compare.items():
    comp_im = functional.convert_image_dtype(torchvision.io.read_image(comp))
    base_im = functional.convert_image_dtype(torchvision.io.read_image(base))
    
    diff_maker = nn.MSELoss(reduction='none')
    diff = torch.sqrt(diff_maker(comp_im, base_im))
    torchvision.io.write_png(functional.convert_image_dtype(diff * 5, torch.uint8), "diff_" + comp)


    criterion = nn.MSELoss()
    loss = torch.sqrt(criterion(comp_im, base_im))



    print('{:<36} {:<36} {:<12}'.format(base, comp, loss))