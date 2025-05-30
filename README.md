# computer-vision-task2
## 对于task1，
首先运行download代码，会下载数据集。然后运行文件中的create代码，自动会生成相关的训练所需要的工具代码，然后运行main函数既可以自动训练。
## 对于task2，
mask rcnn是我自己实现的代码。运行train是在voc数据集上进行训练，运行infer是使用图像进行推理，其中一个是使用测试集内容进行推理，一个是使用外部图像进行推理。
## 对于task2，
sparse rcnn首先下载数据集之后运行voctococo文件，会自动将voc数据集转换成mmdetection可以使用的coco数据集类型。然后git clone mmdetection到本地，将我的config放到mmdetection目录下的configs目录下的sparse-rcnn目录中，运行'''python ./mmdetection/tools/train.py ./sparse-rcnn_r50_fpn_1x_voc.py'''并制定work-dir到对应目录即可。测试的时候只需要运行'''python demo/image_demo.py \
    /root/voc-detection-experiment/data/voc_ins/val07/009764.jpg \
    configs/sparse_rcnn/sparse-rcnn_r50_fpn_1x_voc.py \
    --weights /root/autodl-tmp/model-2/epoch_38.pth \
    --out-dir ./inference_results \
    --pred-score-thr 0.3'''并将图片以及模型权重替换一下即可得到结果。
##
mask rcnn 参数链接：通过网盘分享的文件：final_mask_rcnn_complete.zip
链接: https://pan.baidu.com/s/1mgWJcpV9UXEfaQ3RtFVl3w 提取码: jy2f
sparse rcnn 参数链接：通过网盘分享的文件：epoch_50.zip
链接: https://pan.baidu.com/s/1h0b5O9F4d93r2cr1a45Bow 提取码: ggrm
task1实验模型数量有点多了，参数链接不好整理。老师可以参考我同伴的有关task1的工作。
