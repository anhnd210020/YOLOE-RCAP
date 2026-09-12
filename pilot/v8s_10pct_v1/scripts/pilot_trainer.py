"""Minimal trainer adapter: keep official YOLO dataset path and substitute grounding."""
from ultralytics.data import YOLOConcatDataset, build_yolo_dataset
from ultralytics.data.build import colorstr
from ultralytics.models.yolo.yoloe.train_yoloe_seg import YOLOESegTrainerFromScratch
from ultralytics.utils.torch_utils import de_parallel

from pilot_grounding import PilotGroundingDataset


class PilotYOLOESegTrainerFromScratch(YOLOESegTrainerFromScratch):
    lock_path = None

    def build_dataset(self, img_path, mode="train", batch=None):
        if not self.lock_path:
            raise ValueError("Pilot dataset lock path must be set before training")
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        if mode != "train":
            return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=False, stride=gs, load_vp=False)
        datasets = []
        for item in img_path:
            if isinstance(item, str):
                datasets.append(build_yolo_dataset(self.args, item, batch, self.training_data[item], stride=gs, multi_modal=True))
            else:
                source = item["source_name"]
                if source not in ("GQA", "Flickr30k"):
                    raise ValueError(f"Unknown grounding source: {source}")
                datasets.append(PilotGroundingDataset(
                    img_path=item["img_path"], json_file=item["json_file"],
                    lock_path=self.lock_path, source_name=source, strict_files=True,
                    imgsz=self.args.imgsz, batch_size=batch, augment=True, hyp=self.args,
                    rect=self.args.rect, cache=self.args.cache or None,
                    single_cls=self.args.single_cls or False, stride=int(gs), pad=0.0,
                    prefix=colorstr("train: "), task=self.args.task,
                    classes=self.args.classes, fraction=self.args.fraction,
                    load_vp=self.args.load_vp))
        return YOLOConcatDataset(datasets) if len(datasets) > 1 else datasets[0]


class PilotSmokeTrainer(PilotYOLOESegTrainerFromScratch):
    """Use the real training loop while bypassing forced final-epoch science I/O."""

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        if mode == "val":
            return None  # BaseTrainer still constructs its validator, but never calls it.
        return super().get_dataloader(dataset_path, batch_size=batch_size, rank=rank, mode=mode)

    def validate(self):
        # Pinned BaseTrainer calls validate() on final_epoch despite val=False.
        fitness = -float(self.loss.detach().cpu().item())
        self.best_fitness = fitness
        return self.metrics, fitness

    def save_model(self):
        # Pinned BaseTrainer calls save_model() on final_epoch despite save=False.
        return None

    def final_eval(self):
        # Pinned BaseTrainer calls final_eval() unconditionally after training.
        return None

    def run_callbacks(self, event):
        # The pinned loop emits on_model_save even when save_model is a no-op.
        if event == "on_model_save":
            return None
        return super().run_callbacks(event)
