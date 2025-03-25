from rknn.api import RKNN


class RKNN_model_container:
    def __init__(self, model_path, target=None, device_id=None) -> None:
        rknn = RKNN()

        # Direct Load RKNN Model
        rknn.load_rknn(model_path)

        # print('--> Init runtime environment')
        ret = rknn.init_runtime(target=target, device_id=device_id,async_mode=True,core_mask=RKNN.NPU_CORE_AUTO)
        if ret != 0:
            print('Init runtime environment failed')
            exit(ret)
        # print('done')

        self.rknn = rknn

    def run(self, inputs):
        if self.rknn is None:
            print("ERROR: rknn has been released")
            return []

        if isinstance(inputs, list) or isinstance(inputs, tuple):
            pass
        else:
            inputs = [inputs]

        result = self.rknn.inference(inputs=inputs, data_format="nhwc")

        return result

    def release(self):
        self.rknn.release()
        self.rknn = None
