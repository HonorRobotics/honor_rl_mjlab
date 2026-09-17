"""List registered mjlab environments."""

import mjlab
import mjlab.tasks  # noqa: F401
import tyro
from mjlab.tasks.registry import list_tasks
from prettytable import PrettyTable


def list_environments(keyword: str | None = None) -> int:
    """Print registered environments, optionally filtered by keyword."""
    table = PrettyTable(["#", "Task ID"])
    table.title = "Available Environments in mjlab"
    table.align["Task ID"] = "l"

    matching_tasks = [task_id for task_id in list_tasks() if keyword is None or keyword.lower() in task_id.lower()]
    for index, task_id in enumerate(matching_tasks, start=1):
        table.add_row([index, task_id])

    print(table)
    if not matching_tasks:
        message = "[INFO] No tasks matched"
        if keyword:
            message += f" keyword '{keyword}'"
        print(message)
    return len(matching_tasks)


def main() -> int:
    """Run the command-line interface."""
    return tyro.cli(list_environments, config=mjlab.TYRO_FLAGS)


if __name__ == "__main__":
    main()
