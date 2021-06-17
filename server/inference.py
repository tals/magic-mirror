import torch
import clip

print("Loading clip")
clip_model, clip_preprocess = clip.load("ViT-B/32", jit=False)
print("Loading clip done!")


@torch.no_grad()
def get_image_embedding(img): 
    x = clip_preprocess(img)
    x = clip_model.encode_image(x.cuda()[None])
    x /= x.norm(dim=-1, keepdim=True)
    return x

@torch.no_grad()
def get_text_embedding(classnames):
    zeroshot_weights = []
    for classname in classnames:
        texts = [template.format(classname) for template in imagenet_templates] #format with class
        texts = clip.tokenize(texts).cuda() #tokenize
        class_embeddings = clip_model.encode_text(texts) #embed with text encoder
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        zeroshot_weights.append(class_embedding)
    zeroshot_weights = torch.stack(zeroshot_weights, dim=0).cuda()
    return zeroshot_weights


