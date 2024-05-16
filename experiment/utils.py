import os.path

from PIL import Image

def concat_images(folder):
    origin_folder = os.path.join(folder,"origin")
    recon_folder = os.path.join(folder, "recon")
    mask_folder = os.path.join(folder, "mask")
    gtmask_folder = os.path.join(folder, "gt_mask")
    concat_folder = os.path.join(folder, "concat")

    n = len(os.listdir(origin_folder))
    for i in range(n):
        o = Image.open(os.path.join(origin_folder, f"{i}.png")).resize((256,256))
        r = Image.open(os.path.join(recon_folder, f"{i}.png")).resize((256,256))
        m = Image.open(os.path.join(mask_folder, f"{i}.png")).resize((256,256))
        gtm = Image.open(os.path.join(gtmask_folder, f"{i}.png")).resize((256,256))
        concat = Image.new("RGB", (256*4, 256))
        concat.paste(o, (0,0))
        concat.paste(r, (256+1,0))
        concat.paste(m, (256*2+1,0))
        concat.paste(gtm, (256*3+1,0))
        concat.save(os.path.join(concat_folder, f"v{i}.png"))
def concat_images_v(folder):
    w = 256*4
    h = 256*8
    out = Image.new("RGB", (w, h))
    idx = 0
    for f in os.listdir(folder):
        fn = os.path.join(folder, f)
        im = Image.open(fn)
        out.paste(im, (0, idx*256+1))
        idx += 1
    out.save(os.path.join(folder, "concat.png"))

def inter_image(im1, im2, im3, im4, alpha1=0.2, alpha2=0.5):
    im1 = Image.open(im1).resize((256,256))
    im2 = Image.open(im2).resize((256,256))
    im3 = Image.open(im3).resize((256,256))
    im4 = Image.open(im4).resize((256,256))

    b1 = Image.blend(im1, im2, alpha=alpha1)
    b2 = Image.blend(im3, im4, alpha=alpha2)

    out = Image.blend(b1,b2,alpha=0.5)
    b2.save("blend.png")


if __name__ == '__main__':
    # concat_images("../samples/paper")
    concat_images_v(r"B:\study\毕业设计\论文\论文配图\实验结果\分割结果")
    # inter_image(r"E:\DataSets\AnomalyDetection\mvtec_anomaly_detection\cable\DiffAD\46.jpg",
    #             r"E:\DataSets\AnomalyDetection\mvtec_anomaly_detection\cable\DiffusionAD\46-th.png",
    #             r"E:\DataSets\AnomalyDetection\mvtec_anomaly_detection\cable\draem\46-recon.png",
    #             r"C:\Users\AJ\Downloads\im4.jpg")