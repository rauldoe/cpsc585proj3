
import tensorflow as tf
import timeit

# Testing CUDA
# C:\ProgramData\NVIDIA Corporation\CUDA Samples\v10.2\bin\win64\Debug\deviceQuery.exe
class Hardware:
    @staticmethod
    def checkDevice():
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
        device_name = tf.test.gpu_device_name()
        if device_name != '/device:GPU:0':
            raise SystemError('GPU device not found')
        print('Found GPU at: {}'.format(device_name))

    @staticmethod
    def cpu():
        with tf.device('/cpu:0'):
            random_image_cpu = tf.random.normal((100, 100, 100, 3))
            net_cpu = tf.keras.layers.Conv2D(32, 7)(random_image_cpu)
            return tf.math.reduce_sum(net_cpu)

    @staticmethod
    def gpu():
        with tf.device('/device:GPU:0'):
            random_image_gpu = tf.random.normal((100, 100, 100, 3))
            net_gpu = tf.keras.layers.Conv2D(32, 7)(random_image_gpu)
            return tf.math.reduce_sum(net_gpu)

    @staticmethod
    def testSpeedUp():
        # We run each op once to warm up; see: https://stackoverflow.com/a/45067900
        Hardware.cpu()
        Hardware.gpu()

        # Run the op several times.
        print('Time (s) to convolve 32x7x7x3 filter over random 100x100x100x3 images '
                '(batch x height x width x channel). Sum of ten runs.')
        print('CPU (s):')
        cpu_time = timeit.timeit('cpu()', number=10, setup="from __main__ import cpu")
        print(cpu_time)
        print('GPU (s):')
        gpu_time = timeit.timeit('gpu()', number=10, setup="from __main__ import gpu")
        print(gpu_time)
        print('GPU speedup over CPU: {}x'.format(int(cpu_time/gpu_time)))
