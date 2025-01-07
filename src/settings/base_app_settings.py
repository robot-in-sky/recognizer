import logging
from pathlib import Path
from types import UnionType

from pydantic import BaseModel
from pydantic_core import PydanticUndefined
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.config import ENV_FILE_PATH

_logger = logging.getLogger(__name__)


class BaseAppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE_PATH,
                                      env_file_encoding="utf-8",
                                      env_nested_delimiter="__",
                                      extra="ignore")

    @classmethod
    def get_dotenv_example(cls) -> str:

        def _get_env_field(*, prefix: str, name: str, required: bool, default: str | None) -> str:
            value = "" if required else "# "
            delimiter = cls.model_config["env_nested_delimiter"]
            default = default or ""
            value += f"{prefix.upper()}{delimiter}{name.upper()}={default}"
            return value

        def _get_env_fields_recursive(model_class: type[BaseModel], prefix: str) -> list[str]:
            fields = []
            for name, info in model_class.model_fields.items():
                if (type(info.annotation) is not UnionType and
                        issubclass(info.annotation, BaseModel)):
                    fields.extend(_get_env_fields_recursive(info.annotation, prefix + name))
                else:
                    has_default = info.default is not PydanticUndefined
                    default = str(info.default) if has_default else None
                    field = _get_env_field(prefix=prefix, name=name,
                                       required=info.is_required(), default=default)
                    fields.append(field)
            fields.append("")
            return fields

        env_prefix = cls.model_config.get("env_prefix", "")
        output = _get_env_fields_recursive(cls, env_prefix)
        return "\n".join(output)


    @classmethod
    def save_dotenv_example_if_not_exists(cls) -> None:
        env_file = Path(cls.model_config["env_file"])
        env_file_dir = env_file.parent.resolve()
        if not env_file_dir.exists():
            env_file_dir.mkdir(parents=True)
        example_file_name = env_file.name + ".example"
        example_file = env_file_dir.joinpath(example_file_name)
        if not example_file.exists():
            example_file.write_text(cls.get_dotenv_example(), encoding="utf-8")
            _logger.warning("Dotenv example saved: %s", example_file)
