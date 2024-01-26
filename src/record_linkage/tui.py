from datetime import datetime
from pathlib import Path
from time import sleep
from typing import Iterable

from rich import print
from rich.traceback import Traceback
from textual import on, work
from textual.app import App, ComposeResult, events
from textual.containers import Container, Vertical
from textual.css.query import NoMatches
from textual.screen import Screen
from textual.validation import Number, ValidationResult, Validator
from textual.widget import Widget
from textual.widgets import Button, Footer, Header, Input, Label, RichLog, Static
from textual.worker import get_current_worker

from record_linkage.match import LinkedData
from record_linkage.normalize import NormalizeConfig


class FileExistsValidator(Validator):
    def validate(self, value: str) -> ValidationResult:
        value_path = Path(value)
        if value_path.exists() and not value_path.is_dir():
            return self.success()
        elif value_path.exists() and value_path.is_dir():
            return self.failure(f"File: '{value}' Is a Directory.")
        else:
            return self.failure(f"File: '{value}' Does Not Exist.")


class ValueInput(Container):
    def __init__(
        self,
        *children: Widget,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
        label: str,
        input_id: str,
        tooltip: str | None = None,
        validators: Validator | Iterable[Validator] | None = None,
    ) -> None:
        super().__init__(
            *children, name=name, id=id, classes=classes, disabled=disabled
        )
        self.label = label
        self.input_id = input_id
        self.tooltip = tooltip
        self.validators = validators

    def compose(self) -> ComposeResult:
        label = Label(f"{self.label}:", shrink=True)
        if self.tooltip:
            label.tooltip = self.tooltip
        with Vertical():
            with Container(classes="input-container"):
                yield Container(label, classes="align-right")
                yield Container(
                    Input(
                        placeholder=f"Enter {self.label}",
                        id=self.input_id,
                        validators=self.validators,
                    ),
                    id=f"{self.input_id}-container",
                    classes="input",
                )


class StartingInput(Container):
    BORDER_TITLE = "Starting List Info"

    def compose(self) -> ComposeResult:
        with Vertical():
            yield ValueInput(
                label="File Path",
                input_id="starting-file-path",
                tooltip="Path to Mailing List File (csv)",
                validators=FileExistsValidator(),
            )
            yield ValueInput(
                label="Name Column",
                input_id="starting-name-col",
                tooltip="Field containing the Full Name(s)",
            )
            yield ValueInput(
                label="Address Column",
                input_id="starting-address-col",
                tooltip="Field Containing the Full Address",
            )


class ResultsInput(Container):
    BORDER_TITLE = "Results List Info"

    def compose(self) -> ComposeResult:
        with Vertical():
            yield ValueInput(
                label="File Path",
                input_id="results-file-path",
                tooltip="Path to Results File (csv)",
                validators=FileExistsValidator(),
            )
            yield ValueInput(
                label="Name Column",
                input_id="results-name-col",
                tooltip="Field containing the Full Name(s)",
            )
            yield ValueInput(
                label="Address Column",
                input_id="results-address-col",
                tooltip="Field Containing the Full Address",
            )
            yield ValueInput(
                label="Aggregation Column",
                input_id="results-agg-col",
                tooltip="Field Containing Values to Aggregate.",
            )


class CommonInput(Container):
    BORDER_TITLE = "Misc. Info"

    def compose(self) -> ComposeResult:
        with Vertical():
            yield ValueInput(
                label="ID Column",
                input_id="id-col",
                tooltip="The ID fields in both files must have the same name.",
            )
            yield ValueInput(
                label="Lower Prob. Limit",
                input_id="lower-limit",
                tooltip="The smallest value that"
                + " the probability of record_linkage to accept.",
                validators=Number(minimum=0, maximum=1),
            )
            yield ValueInput(
                label="Output Dir.",
                input_id="output-dir",
                tooltip="Where to write the final matched file.",
            )


class MainScreen(Screen):
    """A Textual app to manage Record Linkage."""

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        with Container(id="main"):
            yield StartingInput(classes="main-container")
            yield ResultsInput(classes="main-container")
            yield CommonInput(classes="main-container")
            yield Container(
                Button("RUN", classes="submit-button"),
                classes="main-container",
            )
        yield Footer()

    @on(Button.Pressed)
    def run_pressed(self, event: Button.Pressed) -> None:
        self.action_match()

    @on(Input.Changed)
    def show_invalid_reasons(self, event: Input.Changed) -> None:
        # Updating the UI to show the reasons why validation failed

        if event.validation_result:
            if not event.validation_result.is_valid:
                try:
                    self.query_one(f"#{event.input.id}-error", Static).update(
                        "/n".join(event.validation_result.failure_descriptions),
                    )
                except NoMatches:
                    self.query_one(f"#{event.input.id}-container").mount(
                        Static(
                            "/n".join(event.validation_result.failure_descriptions),
                            id=f"{event.input.id}-error",
                            classes="error",
                        ),
                    )

            else:
                try:
                    self.query_one(f"#{event.input.id}-error").remove()
                except NoMatches:
                    ...

    @work(exclusive=True, thread=True)
    async def action_match(self):
        def ui_work():
            self.app.push_screen(WorkingScreen())

        worker = get_current_worker()
        if not worker.is_cancelled:
            self.app.call_from_thread(ui_work)

        log = self.app.screen.query_one(OutputLog)
        log.begin_capture_print()
        print("[bold green]Matching In Progress...")

        sleep(5)
        try:
            # TODO: If empty, don't pass into kwargs
            starting_file_path = self.query_one("#starting-file-path", Input).value
            starting_name_col = self.query_one("#starting-name-col", Input).value
            starting_address_col = self.query_one("#starting-address-col", Input).value
            results_file_path = self.query_one("#results-file-path", Input).value
            results_name_col = self.query_one("#results-name-col", Input).value
            results_address_col = self.query_one("#results-address-col", Input).value
            results_agg_col = self.query_one("#results-agg-col", Input).value
            lower_limit = float(self.query_one("#lower-limit", Input).value)
            id_col = self.query_one("#id-col", Input).value
            output_dir = self.query_one("#output-dir", Input).value
            results_config = NormalizeConfig(
                agg_col=results_agg_col,
                id_column=id_col,
                name_col=results_name_col,
                addr_col=results_address_col,
            )
            starting_config = NormalizeConfig(
                id_column=id_col,
                addr_col=starting_address_col,
                name_col=starting_name_col,
            )
            data = LinkedData(
                results_file_path=results_file_path,
                starting_list_path=starting_file_path,
                results_config=results_config,
                starting_list_config=starting_config,
                lower_limit_prob=lower_limit,
            )

            data.match(settings={"unique_id_column_name": id_col})
            out_path = Path(output_dir)
            name = out_path
            if out_path.is_dir():
                if not out_path.exists():
                    out_path.mkdir(parents=True)
                name = out_path / f"Matched - {datetime.now():%Y-%m-%d}.csv"

            data.matched.write_csv(name)
            print(f"\n\n[bold green]You can find the output at {name}")
        except Exception:
            log.write(Traceback(width=None, show_locals=True))

        print("\n\n\n\n")
        print("[bold green]Press Enter to Return...[/]")
        log.end_capture_print()


class OutputLog(RichLog):
    @on(events.Print)
    def on_print(self, event: events.Print) -> None:
        self.write(event.text)


class WorkingScreen(Screen):
    BINDINGS = [
        ("enter", "return_to_main", "Return to the Main Screen"),
    ]

    def action_return_to_main(self):
        self.app.push_screen("main")

    def compose(self) -> ComposeResult:
        yield Header()
        yield OutputLog()
        yield Footer()


class RecordLinkage(App):
    def on_mount(self):
        self.install_screen(MainScreen(), "main")
        self.push_screen("main")


FIELDS = [
    "starting-file-path",
    "starting-name-col",
    "starting-address-col",
    "results-file-path",
    "results-name-col",
    "results-address-col",
    "results-agg-col",
    "id-col",
    "lower-limit",
    "output-dir",
]
# @on(Input.Changed, "#id2")
# async def run_test(self, event: Input.Changed) -> None:
#     self.query_one("#output", Output).text = self.query_one("#id2", Input).value


if __name__ == "__main__":
    app = RecordLinkage(css_path="../../app.tcss")
    app.run()
