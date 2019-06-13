import SimpleITK
from typing import Dict

from evalutils import ClassificationEvaluation
from evalutils.io import ImageIOLoader
from evalutils.validators import (
    NumberOfCasesValidator, UniquePathIndicesValidator, UniqueImagesValidator
)

import metrics_utils as mu

class Glas(ClassificationEvaluation):
    def __init__(self):
        self._metrics_output = {}
        super().__init__(
            file_loader=ImageIOLoader(),
            validators=(
                NumberOfCasesValidator(num_cases=4),
                UniquePathIndicesValidator(),
                UniqueImagesValidator(),
            ),
             aggregates = {
                 "mean",
             },
        )
    
    def evaluate_glas(self):
        self.load()
        self.validate()
        self.merge_ground_truth_and_predictions()
        self.cross_validate()
        self.score()
    
    def save_glas(self):
        self.save()

    def score_case(self, *, idx, case):
        
        gt_path = case["path_ground_truth"]
        pred_path = case["path_prediction"]

        # Load the images for this case
        gt = self._file_loader.load_image(gt_path)
        pred = self._file_loader.load_image(pred_path)

        # Check that they're the right images
        assert self._file_loader.hash_image(gt) == case["hash_ground_truth"]
        assert self._file_loader.hash_image(pred) == case["hash_prediction"]
        
        # Get stats per image
        tp, fp, fn = mu.get_tp_fp_fn(pred, gt) # return true pos, false pos, false neg
        d_temp1, d_temp2 = mu.get_dice_info(pred, gt) # return temp values for first and second terms in obj dice equation
        h_temp1, h_temp2 = mu.get_haus_info(pred, gt) # return temp values for first and second terms in obj hausdorff equation
   
        return {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'd_temp1': d_temp1,
            'd_temp2': d_temp2,
            'h_temp1': h_temp1,
            'h_temp2': h_temp2,
            'pred_fname': pred_path.name,
            'gt_fname': gt_path.name,
        }
        
    
    def metrics_output(self):
        # Get cumulative stats
        tp = self._case_results["tp"].sum()
        fp = self._case_results["fp"].sum()
        fn = self._case_results["fn"].sum()
        d_temp1 = self._case_results["d_temp1"].sum()
        d_temp2 = self._case_results["d_temp2"].sum()
        h_temp1 = self._case_results["h_temp1"].sum()
        h_temp2 = self._case_results["h_temp2"].sum()

        # Calculate object F1
        pr = tp / (tp+fp)
        re = tp / (tp+fn)
        obj_f1 = (2 * pr * re)  / (pr + re)

        # Calculate object dice
        obj_dice = 0.5 * (d_temp1 + d_temp2)
        
        # Calculate object hausdorff
        obj_haus = 0.5 * (h_temp1 + h_temp2)
        
        metric_dict = {'obj_f1': obj_f1, 'obj_dice': obj_dice, 'obj_hausdorff': obj_haus}
        for metric, value in metric_dict.items():
            self._metrics_output[metric] = value
    
    @property
    def _metrics(self) -> Dict:
        return {
            "case": self._case_results.to_dict(),
            "aggregates": self._aggregate_results,
            "metrics": self._metrics_output,
        }

if __name__ == "__main__":
    init = Glas()
    init.evaluate_glas()
    init.metrics_output()
    init.save_glas()

