import asyncio

from webtorch import portable


def test_cached_repository_files_are_one_model_even_in_nested_directories():
    before = portable.webio.list_cache

    async def listed(_cache_dir=None):
        base = "https://huggingface.co/org/model/resolve/main"
        return [
            {"key": base + "/config.json", "size": 10, "total": 10, "complete": True},
            {"key": base + "/encoder/model.safetensors", "size": 20, "total": 20,
             "complete": True},
            {"key": base + "/tokenizer/tokenizer.json", "size": 30, "total": 30,
             "complete": True},
        ]

    portable.webio.list_cache = listed
    try:
        groups = asyncio.run(portable.model_groups())
    finally:
        portable.webio.list_cache = before

    assert len(groups) == 1
    assert groups[0]["label"] == "org/model"
    assert groups[0]["files"] == 3
    assert groups[0]["size"] == 60


class _Step:
    def __init__(self, value=None, done=False):
        self.value = value
        self.done = done


class _Values:
    def __init__(self, values):
        self.values = iter(values)

    async def next(self):
        try:
            return _Step(next(self.values))
        except StopIteration:
            return _Step(done=True)


class _Handle:
    def __init__(self, name, kind="file", children=()):
        self.name = name
        self.kind = kind
        self.children = list(children)

    def values(self):
        return _Values(self.children)


def test_importing_a_multifile_model_means_its_extracted_directory():
    files = [_Handle("config.json"), _Handle("model.safetensors"), _Handle("tokenizer.json")]
    directory = _Handle("my-model", "directory", files)
    before = portable.webio.use_model_file
    seen = []
    portable.webio.use_model_file = lambda handle, name=None: seen.append(name) or name
    try:
        added = asyncio.run(portable.import_model(directory))
    finally:
        portable.webio.use_model_file = before

    assert added == [
        "my-model/config.json",
        "my-model/model.safetensors",
        "my-model/tokenizer.json",
    ]
    assert seen == added


def test_a_zip_is_not_misrepresented_as_an_extracted_model_directory():
    archive = _Handle("model.zip")
    before = portable.webio.use_model_file
    portable.webio.use_model_file = lambda handle, name=None: name or handle.name
    try:
        added = asyncio.run(portable.import_model(archive))
    finally:
        portable.webio.use_model_file = before

    assert added == ["model.zip"]
    assert portable.webio.container_of("model.zip") == ""
