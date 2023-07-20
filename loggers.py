# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
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

import logging
from typing import Any, Dict, Optional

import jax.numpy as jnp
import numpy as np
import omegaconf
from jumanji.training.loggers import Logger
from neptune import new as neptune


class TerminalLogger(Logger):
    """Logs to terminal."""

    def __init__(
        self, name: Optional[str] = None, save_checkpoint: bool = False
    ) -> None:
        super().__init__(save_checkpoint=save_checkpoint)
        if name:
            logging.info(f"Experiment: {name}.")

    def _format_values(self, data: Dict[str, Any]) -> str:
        return " | ".join(
            f"{key.replace('_', ' ').title()}: "
            f"{(f'{value:.3f}' if isinstance(value, (float, jnp.ndarray)) else f'{value:,}')}"
            for key, value in sorted(data.items())
            if np.ndim(value) == 0
        )

    def write(
        self,
        data: Dict[str, Any],
        label: Optional[str] = None,
        env_steps: Optional[int] = None,
    ) -> None:
        if env_steps is not None:
            env_steps_str = f"Env Steps: {env_steps:.2e} | "
        else:
            env_steps_str = ""
        label_str = f"{label.replace('_', ' ').title()} >> " if label else ""
        logging.info(label_str + env_steps_str + self._format_values(data))


class NeptuneLogger(Logger):
    """Logs to the [neptune.ai](https://app.neptune.ai/) platform. The user is expected to have
    their NEPTUNE_API_TOKEN set as an environment variable. This can be done from the Neptune GUI.
    """

    def __init__(
        self,
        name: str,
        project: str,
        cfg: omegaconf.DictConfig,
        save_checkpoint: bool = False,
    ):
        super().__init__(save_checkpoint=save_checkpoint)
        self.run = neptune.init_run(project=project, name=name)
        self.run["config"] = cfg
        self._env_steps = 0.0

    def write(
        self,
        data: Dict[str, Any],
        label: Optional[str] = None,
        env_steps: Optional[float] = None,
    ) -> None:
        if env_steps:
            self._env_steps = env_steps
        prefix = label and f"{label}/"
        for key, metric in data.items():
            if np.ndim(metric) == 0:
                if not np.isnan(metric) and not np.isinf(metric):
                    self.run[f"{prefix}/{key}"].log(
                        float(metric),
                        step=int(self._env_steps),
                        wait=True,
                    )

    def close(self) -> None:
        self.run.stop()

    def upload_checkpoint(self) -> None:
        self.run[f"checkpoint/{self.checkpoint_file_name}"].upload(
            self.checkpoint_file_name
        )
