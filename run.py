import os
from collections import OrderedDict
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer

opt = TestOptions().parse(save=False)
opt.display_id = 0 # do not launch visdom
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.in_the_wild = True # This triggers preprocessing of in the wild images in the dataloader
opt.traverse = True # This tells the model to traverse the latent space between anchor classes
opt.interp_step = 0.05 # this controls the number of images to interpolate between anchor classes

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
visualizer = Visualizer(opt)

# opt.name = 'males_model' # change to 'females_model' if you're trying the code on a female image
opt.name = 'males_model'
model = create_model(opt)
model.eval()

# upload your image (the code supports only a single image at a time)
from google.colab import files
uploaded = files.upload()
for filename in uploaded.keys():
  img_path = filename
  print('User uploaded file "{name}"'.format(name=filename))

data = dataset.dataset.get_item_from_path(img_path)
visuals = model.inference(data)

os.makedirs('results', exist_ok=True)
out_path = os.path.join('results', os.path.splitext(img_path)[0].replace(' ', '_') + '.png')
visualizer.save_images_deploy(visuals, out_path)