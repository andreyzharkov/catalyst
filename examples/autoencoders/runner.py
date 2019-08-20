from typing import Any, Mapping

from torch import nn

from catalyst.dl.core import Runner


class YAERunner(Runner):
    INPUT_IMG_KEY = "images"
    INPUT_Y_KEY = "targets_a"
    INPUT_RANDOM_Y_KEY = "targets_b"

    OUTPUT_IMAGES_A_KEY = "images_a"
    OUTPUT_IMAGES_B_KEY = "images_b"
    OUTPUT_LOGITS_A_KEY = "logits_a"
    OUTPUT_LOGITS_B_KEY = "logits_b"

    OUTPUT_IMPLICIT_LOSS_KEY = "implicit_loss"

    def __init__(
            self,
            model: nn.Module = None,
            device=None,
            input_img_key=INPUT_IMG_KEY,
            input_y_key=INPUT_Y_KEY,
            input_random_y_key=INPUT_RANDOM_Y_KEY,
            output_images_a_key=OUTPUT_IMAGES_A_KEY,
            output_images_b_key=OUTPUT_IMAGES_B_KEY,
            output_logits_a_key=OUTPUT_LOGITS_A_KEY,
            output_logits_b_key=OUTPUT_LOGITS_B_KEY,
            output_implicit_loss_key=OUTPUT_IMPLICIT_LOSS_KEY
    ):
        """

        :param model: model
            with .encoder(x) -> x_explicit, x_implicit
            and  .decoder(y, x_implicit) -> x
        :param device:

        :param input_img_key:
        :param input_y_key:
        :param input_random_y_key:

        :param output_images_a_key:
        :param output_images_b_key:
        :param output_logits_a_key:
        :param output_logits_b_key:
        """
        super().__init__(model=model, device=device)
        self.input_img_key = input_img_key
        self.input_y_key = input_y_key
        self.input_random_y_key = input_random_y_key

        self.input_key = (self.input_img_key, self.input_y_key, self.input_random_y_key)

        self.output_images_a_key = output_images_a_key
        self.output_images_b_key = output_images_b_key
        self.output_logits_a_key = output_logits_a_key
        self.output_logits_b_key = output_logits_b_key

        self.output_implicit_loss_key = output_implicit_loss_key

    def _batch2device(self, batch: Mapping[str, Any], device):
        batch = super()._batch2device(batch, device)
        assert len(batch) == len(self.input_key)
        return dict((k, v) for k, v in zip(self.input_key, batch))

    def predict_batch(self, batch: Mapping[str, Any]):
        images = batch[self.input_img_key]
        targets_a = batch[self.input_y_key]
        targets_b = batch[self.input_random_y_key]

        enc = self.model.encoder
        dec = self.model.decoder
        #
        expl_a, impl_a = enc(images)

        images_a = dec(targets_a, impl_a)
        expl_aa, impl_aa = enc(images_a)

        images_b = dec(targets_b, impl_a)
        expl_ab, impl_ab = enc(images_b)

        impl_loss = ((impl_aa - impl_ab) ** 2).mean()
        return {
            self.output_images_a_key: images_a,
            self.output_images_b_key: images_b,
            self.output_logits_a_key: expl_a,
            self.output_logits_b_key: expl_ab,
            self.output_implicit_loss_key: impl_loss
        }
