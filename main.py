# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from fastapi import FastAPI, File
import uvicorn
from fastai2.vision.all import *
import torch


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def siamese_splitter(model):
    return [params(model.encoder), params(model.head)]

class SiameseImage(fastuple):
    def show(self, ctx=None, **kwargs):
        if len(self) > 2:
            img1, img2, similarity = self
        else:
            img1, img2 = self
            similarity = 'Undetermined'
        if not isinstance(img1, Tensor):
            if img2.size != img1.size: img2 = img2.resize(img1.size)
            t1, t2 = tensor(img1), tensor(img2)
            t1, t2 = t1.permute(2, 0, 1), t2.permute(2, 0, 1)
        else:
            t1, t2 = img1, img2
        line = t1.new_zeros(t1.shape[0], t1.shape[1], 10)
        return show_image(torch.cat([t1, line, t2], dim=2), title=similarity, ctx=ctx, **kwargs)


class SiameseTransform(Transform):
    def __init__(self, files, splits):
        self.valid = {f: self._draw(f) for f in files[splits[1]]}

    def encodes(self, f):
        f2, t = self.valid.get(f, self._draw(f))
        img1, img2 = PILImage.create(f), PILImage.create(f2)
        return SiameseImage(img1, img2, int(t))

    def _draw(self, f):
        same = random.random() < 0.5
        cls = label_func(f)
        if not same: cls = random.choice(L(l for l in labels if l != cls))
        return random.choice(lbl2files[cls]), same


class SiameseModel(Module):
    def __init__(self, encoder, head):
        self.encoder, self.head = encoder, head

    def forward(self, x1, x2):
        ftrs = torch.cat([self.encoder(x1), self.encoder(x2)], dim=1)
        return self.head(ftrs)


class Siamizer():
    def __init__(self, img_base, img_inspect):
        self.img_set = SiameseImage(img_base, img_inspect)

    def siamese_splitter(model):
        return [params(model.encoder), params(model.head)]

    def get_proba(self):
        learner = load_learner('siamese.pkl', cpu=True)

        resp = learner.predict(self.img_set)

        proba = resp[2]

        jsonRet = {
            "chance for similarity": float(proba[1])
        }
        return jsonRet

app = FastAPI()

@app.post("/predict_similarity")
def main(img_base: bytes = File(...), img_inspect: bytes = File(...)):
    local_base_image = Image.open(io.BytesIO(img_base))
    local_inspect_image = Image.open(io.BytesIO(img_inspect))

    siam = Siamizer(local_base_image, local_inspect_image)

    jsonRet = siam.get_proba()

    return jsonRet

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    uvicorn.run("main:app", host='0.0.0.0', port=8123, reload=True)

