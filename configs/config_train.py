from easydict import EasyDict as edict

config = edict()
# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------

# Recognition val targets
config.val_targets = ['lfw', 'cfp_fp', "agedb_30", 'calfw', 'cplfw']

# Recognition data
config.rec = ""  #### Path for the training dataset
config.num_classes = 85742
# config.num_image = 5822653
config.num_image = 600000
# Analysis data


# Data loading settings
config.img_size = 112
config.batch_size = 128
config.recognition_bz = 128
config.age_gender_bz = 128
config.CelebA_bz = 128
config.expression_bz = 128
config.hwb_bz = 128
config.train_num_workers = 2
config.train_pin_memory = True

config.val_batch_size = 256
config.val_num_workers = 0
config.val_pin_memory = True

# Data argument

config.INTERPOLATION = 'bicubic'
config.RAF_NUM_CLASSES = 7
# Label Smoothing
config.RAF_LABEL_SMOOTHING = 0.1

config.AUG_COLOR_JITTER = 0.3
# Use AutoAugment policy. "v0" or "original"
config.AUG_AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
# Random erase prob
config.AUG_REPROB = 0.15
# Random erase mode
config.AUG_REMODE = 'pixel'
# Random erase count
config.AUG_RECOUNT = 1
# Mixup alpha, mixup enabled if > 0
config.AUG_MIXUP = 0.0 #0.8
# Cutmix alpha, cutmix enabled if > 0
config.AUG_CUTMIX = 0.0 #1.0
# Cutmix min/max ratio, overrides alpha and enables cutmix if set
config.AUG_CUTMIX_MINMAX = None
# Probability of performing mixup or cutmix when either/both is enabled
config.AUG_MIXUP_PROB = 0.4
# Probability of switching to cutmix when both mixup and cutmix enabled
config.AUG_MIXUP_SWITCH_PROB = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
config.AUG_MIXUP_MODE = 'batch'

config.AUG_SCALE_SET = True
config.AUG_SCALE_SCALE = (1.0, 1.0)
config.AUG_SCALE_RATIO = (1.0, 1.0)

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------

config.network = "swin_t"

config.fam_kernel_size=3
config.fam_in_chans=2112
config.fam_conv_shared=False
config.fam_conv_mode="split"
config.fam_channel_attention="ECA"
config.fam_spatial_attention="Coord"
config.fam_pooling="max"
config.fam_la_num_list=[2 for j in range(4)]
config.fam_feature="all"
config.fam = "3x3_2112_F_s_C_N_max"

config.embedding_size = 512

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------

# Resume and init
config.resume = False
config.resume_step = 0
config.init = True
config.init_model = "/d9lab/tulei/SwinFace"

# Step num
config.warmup_step = 7500
config.total_step = 80000

# SGD optimizer
#config.optimizer = "sgd"
#config.lr = 0.1
#config.momentum = 0.9
#config.weight_decay = 5e-4

# AdamW optimizer
config.optimizer = "adamw"
config.lr = 5e-4
config.weight_decay = 0.05

# Learning rate
config.lr_name = 'cosine'
config.warmup_lr = 1e-6
config.min_lr = 1e-6
config.decay_epoch = 10 # Epoch interval to decay LR, used in StepLRScheduler
config.decay_rate = 0.1 # LR decay rate, used in StepLRScheduler

# Recognition loss
config.margin_list = (1.0, 0.0, 0.4)
config.sample_rate = 0.3 # Partial FC
config.interclass_filtering_threshold = 0 # Partial FC

# Loss weight
config.recognition_loss_weight = 1.0
config.analysis_loss_weights = [1.0, 0.4, 1.1, 1.5] # Age,Gender,Expersions, BMI

# Others
config.fp16 = True
config.dali = False # For Large Sacle Dataset, such as WebFace42M
config.seed = 2048

# -----------------------------------------------------------------------------
# Output and Saving
# -----------------------------------------------------------------------------

config.save_all_states = True
config.output = "" ####Path for Output

config.verbose = 2000
config.save_verbose = 4000
config.frequent = 40

