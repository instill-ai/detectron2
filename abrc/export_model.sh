python ../tools/deploy/export_model.py \
--config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
--format torchscript --export-method scripting \
--output export/stomata200-mix/cpu \
--sample-image ../../google-drive/stomata200-mix/images/0611_01-01.3.4.tif \
--device cpu \
MODEL.WEIGHTS output/stomata200-mix_output_ep32000_2023_02_14_19_32_50/train_output/model_final.pth \
MODEL.DEVICE cpu \
MODEL.ROI_HEADS.NUM_CLASSES 1 \
MODEL.ROI_HEADS.SCORE_THRESH_TEST 0.3


# Export stomata100 for CPU

#python ../tools/deploy/export_model.py \
#--config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
#--format torchscript --export-method scripting \
#--output export/stomata100 \
#--sample-image ../../google-drive/stomata100/images/0611_01-01.3.4.tif \
#--device cpu \
#MODEL.WEIGHTS output/stomata100_output_ep32000_2023_02_13_20_58_25/train_output/model_final.pth \
#MODEL.DEVICE cpu \
#MODEL.ROI_HEADS.NUM_CLASSES 1 \
#MODEL.ROI_HEADS.SCORE_THRESH_TEST 0.3


# Export stomata100-museum for CPU

#python ../tools/deploy/export_model.py \
#--config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
#--format torchscript --export-method scripting \
#--output export/stomata100-museum/cpu \
#--sample-image ../../google-drive/stomata100-museum/images/1_c_tr_6_cut_want.jpg \
#--device cpu \
#MODEL.WEIGHTS output/stomata100-museum_output_ep32000_2023_02_14_17_02_40/train_output/model_final.pth \
#MODEL.DEVICE cpu \
#MODEL.ROI_HEADS.NUM_CLASSES 1 \
#MODEL.ROI_HEADS.SCORE_THRESH_TEST 0.3


# Export stomata200-mix

#python ../tools/deploy/export_model.py \
#--config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
#--format torchscript --export-method scripting \
#--output export/stomata200-mix/cpu \
#--sample-image ../../google-drive/stomata100/images/0611_01-01.3.4.tif \
#--device cpu \
#MODEL.WEIGHTS output/stomata200-mix_output_ep32000_2023_02_14_19_32_50/train_output/model_final.pth \
#MODEL.DEVICE cpu \
#MODEL.ROI_HEADS.NUM_CLASSES 1 \
#MODEL.ROI_HEADS.SCORE_THRESH_TEST 0.3