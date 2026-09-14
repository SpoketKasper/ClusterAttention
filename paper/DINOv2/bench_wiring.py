import timm

from cluster_attention import ModifiedModel
from paper.bench_utils import ModelLoader, BenchDataLoader
from paper.DINOv2.attention_wiring import ModifiedViTAttention

class ViTModelLoader(ModelLoader):
    wrapper_classes = [ModifiedViTAttention]
    res = None
    model_name = None

    def __init__(self, res, model_name):
        self.res = res
        self.model_name = model_name

    def __call__(self):
        model = timm.create_model(self.model_name, pretrained=True, img_size=self.res)
        model.eval().cuda().bfloat16()
        return model

    def forward(self, model, x):
        return model.forward_features(x)

class ViTImageLoader(BenchDataLoader):
    def __init__(self, res, image=None, ref_res=5306):
        self.res = res
        self.ref_res = ref_res
        self.image = image  # gets set, no counter!

    def __call__(self):
        from PIL import Image
        from timm.data import resolve_data_config, create_transform
        Image.MAX_IMAGE_PIXELS = None
        if self.image in ("deer", "hamngatan"):
            img = Image.open(f"/data/{self.image}.jpg")
        else:  # div8k
            import glob, random
            paths = sorted(glob.glob(f"/images/div8k/*"))
            #print(paths[:3])
            #random.Random(0).shuffle(paths)
            random.Random(42).shuffle(paths)
            #print(paths[:3])
            img = Image.open(paths[self.image])
        config = resolve_data_config({"input_size": (3, self.res, self.res)})
        transform = create_transform(**config, is_training=False)
        return transform(img)[None].cuda().bfloat16()
