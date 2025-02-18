from graphfusionai.tools.base import BaseTool
from graphfusionai.tools.registry import ToolRegistry
import os
from pathlib import Path
from typing import Optional, Union, BinaryIO
import shutil

class FileTool(BaseTool):
    """
    Advanced tool for file operations including reading, writing, copying, and managing files.
    Supports both text and binary files with various modes and encoding options.
    """
    def __init__(self):
        super().__init__(
            name="file",
            description="Advanced file operations tool for reading, writing and managing files"
        )

    def execute(self, 
                action: str,
                file_path: Union[str, Path],
                content: Optional[Union[str, bytes]] = None,
                mode: str = 'r',
                encoding: str = 'utf-8',
                dest_path: Optional[Union[str, Path]] = None,
                create_dirs: bool = False) -> Union[str, bytes]:
        """
        Execute file operations with advanced functionality.

        Args:
            action: Operation to perform ('read', 'write', 'append', 'copy', 'delete', 'exists')
            file_path: Path to the target file
            content: Content to write (for write/append operations)
            mode: File mode ('r', 'rb', 'w', 'wb', 'a', 'ab')
            encoding: File encoding for text operations
            dest_path: Destination path for copy operations
            create_dirs: Create parent directories if they don't exist

        Returns:
            Operation result (file content for read operations, status message for others)

        Raises:
            ValueError: For invalid actions or parameters
            FileNotFoundError: When file operations fail
            PermissionError: When permission is denied
        """
        file_path = Path(file_path)
        
        if create_dirs and action in ('write', 'append'):
            file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if action == "read":
                with open(file_path, mode=mode, encoding=encoding if 'b' not in mode else None) as f:
                    return f.read()

            elif action == "write":
                if content is None:
                    raise ValueError("Content must be provided for write operation")
                with open(file_path, mode=mode, encoding=encoding if 'b' not in mode else None) as f:
                    f.write(content)
                return f"Successfully wrote to {file_path}"

            elif action == "append":
                if content is None:
                    raise ValueError("Content must be provided for append operation")
                with open(file_path, mode='a' if 'b' not in mode else 'ab', 
                         encoding=encoding if 'b' not in mode else None) as f:
                    f.write(content)
                return f"Successfully appended to {file_path}"

            elif action == "copy":
                if not dest_path:
                    raise ValueError("Destination path required for copy operation")
                dest_path = Path(dest_path)
                if create_dirs:
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, dest_path)
                return f"Successfully copied {file_path} to {dest_path}"

            elif action == "delete":
                if file_path.exists():
                    file_path.unlink()
                    return f"Successfully deleted {file_path}"
                return f"File {file_path} does not exist"

            elif action == "exists":
                return str(file_path.exists())

            else:
                raise ValueError(f"Unsupported action: {action}")

        except (FileNotFoundError, PermissionError, OSError) as e:
            return f"Operation failed: {str(e)}"

# Register the tool
registry = ToolRegistry()
registry.register(FileTool)