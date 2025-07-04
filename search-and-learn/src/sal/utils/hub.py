#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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


from typing import List

from huggingface_hub import list_repo_refs, repo_exists


def get_dataset_revisions(dataset_id: str) -> List[str]:
    """Get the list of revisions for a dataset on the Hub."""
    if not repo_exists(dataset_id, repo_type="dataset"):
        return []
    refs = list_repo_refs(dataset_id, repo_type="dataset")
    return [ref.name for ref in refs.branches if ref.name != "main"]
