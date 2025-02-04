import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic_core import PydanticUndefined
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.config import ENV_FILE_PATH

_logger = logging.getLogger(__name__)


class AppSettingsBase(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE_PATH,
                                      env_file_encoding="utf-8",
                                      env_nested_delimiter="__",
                                      extra="ignore")

    @classmethod
    def get_dotenv_example(cls) -> str:

        def _get_env_field(*, prefix: str, name: str, required: bool, default: Any) -> str:
            output = "" if required else "# "
            output += f"{prefix.upper()}{name.upper()}="
            if default is not None:
                output += f"{default}"
            return output

        def _get_env_fields_recursive(model_class: type[BaseModel], prefix: str) -> list[str]:
            lines = []
            for name, info in model_class.model_fields.items():
                if isinstance(info.annotation, type) and issubclass(info.annotation, BaseModel):
                    delimiter = cls.model_config["env_nested_delimiter"]
                    _prefix = f"{prefix}{name}{delimiter}"
                    lines.extend(_get_env_fields_recursive(info.annotation, _prefix))
                else:
                    has_default = info.default is not PydanticUndefined
                    default = info.default if has_default else None
                    field_output = _get_env_field(prefix=prefix, name=name,
                                       required=info.is_required(), default=default)
                    lines.append(field_output)
            lines.append("")
            return lines

        env_prefix = cls.model_config.get("env_prefix", "")
        result = _get_env_fields_recursive(cls, env_prefix)
        return "\n".join(result)


    @classmethod
    def save_dotenv_example(cls, *, exist_ok: bool = True) -> None:
        env_file = Path(cls.model_config["env_file"])
        env_file_dir = env_file.parent.resolve()
        if not env_file_dir.exists():
            env_file_dir.mkdir(parents=True)
            _logger.warning("Dotenv directory created: %s", env_file_dir)
        example_file_name = env_file.name + ".example"
        example_file = env_file_dir.joinpath(example_file_name)
        if not example_file.exists() or exist_ok:
            example_file.write_text(cls.get_dotenv_example(), encoding="utf-8")
            _logger.warning("Dotenv example file saved: %s", example_file)
