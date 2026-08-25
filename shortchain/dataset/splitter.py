"""Group-aware splitting.

Preserves task boundaries so all rows from the same task remain in the
same split or fold, preventing data leakage.

Note:
    The class name is retained for backward compatibility. Current
    implementation does not yet perform stratified splitting.

Future versions may add stratification using metadata such as app name
or tool-count buckets.

Why grouping matters
--------------------
Every positive/negative row belongs to a task; if a task's rows reach both
train and validation, the model can effectively memorise the scenario and
validation looks optimistic. For per-decision (span) use the same guarantee
applies at the task level: all of a task's decisions stay in one fold, so the
"state" features — built only from steps before each decision — are always
out-of-fold for the model that scores them.
"""

from __future__ import annotations

from typing import Iterator

import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from shortchain.config import SplitterConfig
from shortchain.utils.logging import get_logger

log = get_logger(__name__)


class GroupStratifiedSplitter:
    """Group-aware splitter that respects task boundaries.

    Parameters
    ----------
    config
        Splitter configuration.
    """

    def __init__(self, config: SplitterConfig | None = None) -> None:
        self.config = config or SplitterConfig()

    def train_test_split(
        self,
        df: pd.DataFrame,
        random_state: int = 42,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split into train / test, keeping task groups intact.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            ``(train_df, test_df)``
        """
        group_col = self.config.group_by
        if group_col not in df.columns:
            raise ValueError(f"Group column '{group_col}' not found in DataFrame")

        groups = df[group_col]
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=self.config.test_size,
            random_state=random_state,
        )

        train_idx, test_idx = next(splitter.split(df, groups=groups))
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        log.info(
            f"Train/test split: [bold green]{len(train_df)}[/bold green] train, "
            f"[bold yellow]{len(test_df)}[/bold yellow] test "
            f"({train_df[group_col].nunique()} / {test_df[group_col].nunique()} tasks)"
        )
        return train_df, test_df

    def kfold_split(
        self,
        df: pd.DataFrame,
    ) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
        """Yield (train, val) pairs for k-fold cross-validation.

        Groups are respected: all rows from one task stay in the same
        fold.

        Yields
        ------
        tuple[pd.DataFrame, pd.DataFrame]
            ``(train_fold, val_fold)``
        """
        group_col = self.config.group_by
        if group_col not in df.columns:
            raise ValueError(f"Group column '{group_col}' not found in DataFrame")

        groups = df[group_col]
        kfold = GroupKFold(n_splits=self.config.n_folds)

        for fold_idx, (train_idx, val_idx) in enumerate(
            kfold.split(df, groups=groups)
        ):
            train_fold = df.iloc[train_idx].reset_index(drop=True)
            val_fold = df.iloc[val_idx].reset_index(drop=True)
            log.info(
                f"Fold {fold_idx + 1}/{self.config.n_folds}: "
                f"{len(train_fold)} train, {len(val_fold)} val "
                f"({train_fold[group_col].nunique()}/{val_fold[group_col].nunique()} tasks)"
            )
            yield train_fold, val_fold
