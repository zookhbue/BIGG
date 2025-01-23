The proposed BIGG model combines the diffusion model and the generative adversarial networks.



(1) introduction

ckpt/trained_model.pth.tar is the trained model

data/*.txt are the training data. 
data/example_empirical.txt is an example of empirical time-series, which is derived from the four-dimensional fMRI by using the GRETNA software.   
data/example_coarse.txt is an example of coarse time-series, which is derived from the four-dimensional fMRI by our nonparametric method (using the fmri_parcellation.py).


fmri_parcellation.py can transform the original four-dimensional fMRI into two-dimensional coarse time-series without any learning pamameters.


model/loss.py  is the defined loss functions, including the generative loss, the discriminative loss, the reconstructed loss, the sparse connective penalty loss, and the classification loss.


model/blocks.py defines basic calculation function, including LayerNorm, ConvNorm, Downsample, Upsample, SeTeBlock, Attention. These functions are used to construct the BrainNetCNN, Denoiser modules.


model/modules.py defines the BrainNetCNN classifier, and the Denoiser (generator) networks.


model/diffusion.py is the classical diffusion model, including the diffusion process and the denoising process.


model/diffGANmodel.py defines the generator and the discriminator.


dataset.py defines how data is read.


tools.py defines the training strategy, i.e., the optimization method, the learning rate.

test.txt has three columns: empirical time-series file, coarse time-series file, label. It use the second column to generate the clean time-series and the brain effective connectivity.

demo.py is an example to generate the brain effective connectivity. 


(2) how to run
==> if you use the command window
pip install -r requirements.txt
python3 demo.py

==> if you use the pycharm
open the demo.py file, and then run it.


the output is the predicted label (shown in the command window) and the brain effective connectivity (saved in directory: output/example.jpg).

















