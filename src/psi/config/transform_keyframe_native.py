"""Keep the current Psi0 observation path while attaching native-grid memory."""
from psi.config.transform_framesamp import FrameSampDataTransform


class NativeKeyframeDataTransform(FrameSampDataTransform):
    def __call__(self, data, **kwargs):
        bank = kwargs.pop("native_keyframe_bank", None)
        if bank is None:
            raise RuntimeError("NativeKeyframeDataTransform requires a verified native bank")
        if "framesamp_feature_bank" in kwargs:
            raise ValueError("Native memory cannot also receive a legacy feature bank")
        return super().__call__(data, framesamp_feature_bank=bank, **kwargs)
