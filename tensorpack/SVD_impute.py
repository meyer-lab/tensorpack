# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from tensorly import partial_svd
import numpy as np


class IterativeSVD(object):
    def __init__(
            self,
            rank,
            convergence_threshold=1e-7,
            max_iters=500,
            random_state=None,
            min_value=None,
            max_value=None,
            verbose=False):
        self.min_value=min_value
        self.max_value=max_value
        self.rank = rank
        self.max_iters = max_iters
        self.convergence_threshold = convergence_threshold
        self.verbose = verbose
        self.random_state = random_state

    def clip(self, X):
        """
        Clip values to fall within any global or column-wise min/max constraints
        """
        X = np.asarray(X)
        if self.min_value is not None:
            X[X < self.min_value] = self.min_value
        if self.max_value is not None:
            X[X > self.max_value] = self.max_value
        return X

    def prepare_input_data(self, X):
        """
        Check to make sure that the input matrix and its mask of missing
        values are valid. Returns X and missing mask.
        """
        if X.dtype != "f" and X.dtype != "d":
            X = X.astype(float)

        assert X.ndim == 2
        missing_mask = np.isnan(X)
        assert not missing_mask.all()
        return X, missing_mask

    def fit_transform(self, X, y=None):
        """
        Fit the imputer and then transform input `X`
        Note: all imputations should have a `fit_transform` method,
        but only some (like IterativeImputer in sklearn) also support inductive
        mode using `fit` or `fit_transform` on `X_train` and then `transform`
        on new `X_test`.
        """
        X_original, missing_mask = self.prepare_input_data(X)
        observed_mask = ~missing_mask
        X_filled = X_original.copy()
        X_filled[missing_mask] = 0.0
        assert isinstance(X_filled, np.ndarray)
        X_result = self.solve(X_filled, missing_mask)
        assert isinstance(X_result, np.ndarray)
        X_result = self.clip(np.asarray(X_result))
        X_result[observed_mask] = X_original[observed_mask]
        return X_result

    def _converged(self, X_old, X_new, missing_mask):
        F32PREC = np.finfo(np.float32).eps
        # check for convergence
        old_missing_values = X_old[missing_mask]
        new_missing_values = X_new[missing_mask]
        difference = old_missing_values - new_missing_values
        ssd = np.sum(difference ** 2)
        old_norm_squared = (old_missing_values ** 2).sum()
        # edge cases
        if old_norm_squared == 0 or \
                (old_norm_squared < F32PREC and ssd > F32PREC):
            return False
        else:
            return (ssd / old_norm_squared) < self.convergence_threshold

    def solve(self, X, missing_mask):
        observed_mask = ~missing_mask
        X_filled = X
        for i in range(self.max_iters):
            curr_rank = self.rank
            self.U, S, V = partial_svd(X_filled, curr_rank, random_state=self.random_state)
            X_reconstructed = self.U @ np.diag(S) @ V
            X_reconstructed = self.clip(X_reconstructed)

            # Masked mae
            mae = np.mean(np.abs(X[observed_mask] - X_reconstructed[observed_mask]))

            if self.verbose:
                print(
                    "[IterativeSVD] Iter %d: observed MAE=%0.6f" % (
                        i + 1, mae))
            converged = self._converged(
                X_old=X_filled,
                X_new=X_reconstructed,
                missing_mask=missing_mask)
            X_filled[missing_mask] = X_reconstructed[missing_mask]
            if converged:
                break
        return X_filled
