## Codebase for paper: Analyzing the Hidden Activations of Deep Policy Networks: Why Representation Matters

### Training VAEs
Run `./train_vae.py` and alter the following attributes to fit your usecase:
* KukaEnv(...)
    * `static_all` : _static, static_ task
    * `static_obj_rnd_pos` : _static, random_ task
* VAE(...)
    * Enter an `int` that corresponds to the dimensions of the latent vector
    
### Training full-image agents
Run `simulate.py --episodes 25000 --name <name> --seed <seed> --images` and alter the following attributes to fit your usecase:
* KukaEnv(...)
    * `static_all` : _static, static_ task
    * `static_obj_rnd_pos` : _static, random_ task
    
### Training latent agents
Run `simulate_encoder.py --episodes 25000 --z-dims <latent dim> --name <name> --seed <seed> --images` and alter the following attributes to fit your usecase:
* KukaEnv(...)
    * `static_all` : _static, static_ task
    * `static_obj_rnd_pos` : _static, random_ task
    
    
### Creating PCA plots
To create the plots seen in the paper, use `pca_plots.ipynb`

Depending on the model-task combination, you will need to alter:
* `Z_DIMS`
* The `.pth` file for the VAE
* The `.pth` files for:
    * `model` : main model driving the state-reward pair collection process
    * `plotting_model` : load your latent weight snapshots here
    * `model_cnn` : load your image weight snapshots here