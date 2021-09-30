from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import threading
import numpy as np
import picamera

from PIL import Image
from tflite_runtime.interpreter import Interpreter

class TensorCam(threading.Thread):
  m_bPressed = False
  m_rResult ={}
  m_bExit = False
  
  def __init__(self):
    super().__init__()

  def start_recognize(self, bEnable) :
      self.m_bPressed = bEnable

  def load_labels(self, path):
    with open(path, 'r') as f:
      return {i: line.strip() for i, line in enumerate(f.readlines())}

  def set_input_tensor(self, interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image

  def get_result(self):
    s = self.m_rResult.keys()
    return list(s)

  def classify_image(self, interpreter, image, top_k=1):
    """Returns a sorted array of classification results."""
    self.set_input_tensor(interpreter, image)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    output = np.squeeze(interpreter.get_tensor(output_details['index']))

    # If the model is quantized (uint8 data), then dequantize the results
    if output_details['dtype'] == np.uint8:
      scale, zero_point = output_details['quantization']
      output = scale * (output - zero_point)

    ordered = np.argpartition(-output, top_k)
    return [(i, output[i]) for i in ordered[:top_k]]

  def exit(self):
    self.m_bExit = True

  def run(self):

    labels = self.load_labels('labels.txt')
    interpreter = Interpreter('model_unquant.tflite')

    interpreter.allocate_tensors()
    _, height, width, _ = interpreter.get_input_details()[0]['shape'] 

    with picamera.PiCamera(resolution=(640, 480), framerate=3) as camera:
      camera.start_preview(fullscreen=False, window=(100,100,640,480) )
      try:
        stream = io.BytesIO()
        for _ in camera.capture_continuous(
          stream, format='jpeg', use_video_port=True):
      
          stream.seek(0)
          image = Image.open(stream).convert('RGB').resize((width, height),Image.ANTIALIAS)
          results = self.classify_image(interpreter, image)

          label_id, prob = results[0]

          stream.seek(0)
          stream.truncate()
          camera.annotate_text = '%s %.0f%%' % (labels[label_id], prob*100 )

          if self.m_bPressed :
            self.m_rResult[labels[label_id].split()[1]]=label_id
            self.m_bPressed = False
            camera.annotate_text = '%s added.' % (labels[label_id] )

          if self.m_bExit:
            break
              
      finally:
        camera.stop_preview()


