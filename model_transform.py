import os, sys
import tensorflow as tf
from models.FS_model import FS_UNET


# 恢复模型
fs_model = FS_UNET()
checkpoint_dir = 'training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(fs_model=fs_model)
ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)

# 如果存在检查点，恢复最新版本检查点
if ckpt_manager.latest_checkpoint:
  checkpoint.restore(ckpt_manager.latest_checkpoint)

# 保存为其它格式模型文件
fs_model.save('Model/h5/face_parse_test.h5')
tf.saved_model.save(fs_model, 'Model/pb/face_parse_test')

# 加载模型
# model_pb = tf.saved_model.load('Model/pb/face_parse')
model_h5 = tf.keras.models.load_model('Model/h5/face_parse_test.h5')

# 指定输入shape, 通过pb模型恢复
# concrete_func = model.signatures[
#   tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
# concrete_func.inputs[0].set_shape([1, 640, 640, 3])


# 转换成tflite
converter = tf.lite.TFLiteConverter.from_keras_model(model_h5)
# converter = tf.lite.TFLiteConverter.from_saved_model('Model/pb/face_parse')
# converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
# converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_types = [tf.lite.FLOAT16]
converter.post_training_quantize = True
tflite_model = converter.convert()


# 存储
open("Model/tflite/face_parse_test.tflite", "wb").write(tflite_model)
