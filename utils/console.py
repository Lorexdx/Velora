try:
    from colorama import just_fix_windows_console

    just_fix_windows_console()
except ImportError:
    pass

class ConsoleStyle:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    MAGENTA = "\033[95m"
    RED = "\033[91m"

def color(text, style):
    return f"{style}{text}{ConsoleStyle.RESET}"

def format_device(device):
    label = color("Urzadzenie:", ConsoleStyle.CYAN)
    value = color(str(device), ConsoleStyle.BOLD)

    return f"{label} {value}"

def format_training_step(step, loss, learning_rate=None, tokens_per_second=None):
    step_label = color("Etap:", ConsoleStyle.CYAN)
    step_value = color(step, ConsoleStyle.GREEN)
    loss_label = color("Strata:", ConsoleStyle.MAGENTA)
    loss_value = color(f"{loss:.4f}", ConsoleStyle.YELLOW)
    message = f"{step_label} {step_value} | {loss_label} {loss_value}"

    if learning_rate is not None:
        lr_value = color(f"{learning_rate:.2e}", ConsoleStyle.CYAN)
        message = f"{message} | lr: {lr_value}"

    if tokens_per_second is not None:
        speed_value = color(f"{tokens_per_second:,.0f}", ConsoleStyle.GREEN)
        message = f"{message} | tok/s: {speed_value}"

    return message


def format_evaluation_step(step, train_loss, val_loss, best_val_loss=None):
    step_label = color("Ewaluacja:", ConsoleStyle.CYAN)
    step_value = color(step, ConsoleStyle.GREEN)
    train_value = color(f"{train_loss:.4f}", ConsoleStyle.YELLOW)
    val_value = color(f"{val_loss:.4f}", ConsoleStyle.MAGENTA)
    message = (
        f"{step_label} {step_value} | "
        f"train_loss: {train_value} | val_loss: {val_value}"
    )

    if best_val_loss is not None:
        best_value = color(f"{best_val_loss:.4f}", ConsoleStyle.GREEN)
        message = f"{message} | best_val_loss: {best_value}"

    return message


def format_checkpoint_saved(path, step, loss, is_best=False):
    kind = "Best checkpoint" if is_best else "Checkpoint"
    label = color(f"{kind}:", ConsoleStyle.GREEN)
    details = f"step={step}, loss={loss:.4f}"

    return f"{label} {path} ({details})"


def format_checkpoint_loaded(path, step, loss):
    label = color("Wczytano checkpoint:", ConsoleStyle.CYAN)
    details = f"step={step}, loss={loss:.4f}"

    return f"{label} {path} ({details})"


def format_checkpoint_skipped(reason):
    label = color("Pominieto checkpoint:", ConsoleStyle.YELLOW)

    return f"{label} {reason}"


def format_dataset_cache_loaded(path, token_count):
    label = color("Wczytano cache tokenow:", ConsoleStyle.CYAN)

    return f"{label} {path} ({token_count:,} tokenow)"


def format_dataset_cache_saved(path, token_count):
    label = color("Zapisano cache tokenow:", ConsoleStyle.GREEN)

    return f"{label} {path} ({token_count:,} tokenow)"


def format_dataset_tokenized(token_count):
    label = color("Tokenow:", ConsoleStyle.CYAN)

    return f"{label} {token_count:,}"


def format_training_interrupted():
    return color("Przerwano trening. Zapisuje checkpoint...", ConsoleStyle.YELLOW)
