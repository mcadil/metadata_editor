"""
Ebook Metadata Editor
A tkinter GUI application for editing metadata in PDF and EPUB files.
Follows SOLID principles for clean, maintainable architecture.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import subprocess
import json
import os
import re


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class BookMetadata:
    """Data class representing book metadata."""

    title: str = ""
    author: str = ""
    subject: str = ""
    keywords: str = ""

    def copy(self) -> "BookMetadata":
        return BookMetadata(
            title=self.title,
            author=self.author,
            subject=self.subject,
            keywords=self.keywords,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BookMetadata):
            return False
        return (
            self.title == other.title
            and self.author == other.author
            and self.subject == other.subject
            and self.keywords == other.keywords
        )


@dataclass
class BookFile:
    """Represents a loaded book file with its metadata."""

    path: Path
    original_metadata: BookMetadata = field(default_factory=BookMetadata)
    current_metadata: BookMetadata = field(default_factory=BookMetadata)
    error: Optional[str] = None

    @property
    def is_modified(self) -> bool:
        return self.original_metadata != self.current_metadata

    @property
    def filename(self) -> str:
        return self.path.name

    @property
    def extension(self) -> str:
        return self.path.suffix.lower()


@dataclass
class ExternalApp:
    """Configuration for an external application."""

    name: str
    path: str
    arguments: str = "{file}"  # {file} will be replaced with the file path
    file_types: List[str] = field(default_factory=lambda: [".pdf", ".epub"])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "arguments": self.arguments,
            "file_types": self.file_types,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ExternalApp":
        return ExternalApp(
            name=data.get("name", ""),
            path=data.get("path", ""),
            arguments=data.get("arguments", "{file}"),
            file_types=data.get("file_types", [".pdf", ".epub"]),
        )


class ExternalAppManager:
    """Manages external application configurations."""

    CONFIG_FILE = "external_apps.json"

    def __init__(self):
        self._apps: List[ExternalApp] = []
        self._config_path = Path.home() / ".ebook_metadata_editor" / self.CONFIG_FILE
        self._load_config()

    def _load_config(self):
        """Load configuration from file."""
        try:
            if self._config_path.exists():
                with open(self._config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._apps = [
                        ExternalApp.from_dict(app) for app in data.get("apps", [])
                    ]
        except Exception:
            self._apps = []

    def _save_config(self):
        """Save configuration to file."""
        try:
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._config_path, "w", encoding="utf-8") as f:
                json.dump({"apps": [app.to_dict() for app in self._apps]}, f, indent=2)
        except Exception as e:
            print(f"Failed to save config: {e}")

    @property
    def apps(self) -> List[ExternalApp]:
        return self._apps.copy()

    def add_app(self, app: ExternalApp):
        self._apps.append(app)
        self._save_config()

    def remove_app(self, index: int):
        if 0 <= index < len(self._apps):
            del self._apps[index]
            self._save_config()

    def update_app(self, index: int, app: ExternalApp):
        if 0 <= index < len(self._apps):
            self._apps[index] = app
            self._save_config()

    def move_app(self, index: int, direction: int):
        """Move app up (-1) or down (+1) in the list."""
        new_index = index + direction
        if 0 <= new_index < len(self._apps):
            self._apps[index], self._apps[new_index] = (
                self._apps[new_index],
                self._apps[index],
            )
            self._save_config()

    def get_apps_for_file(self, file_path: Path) -> List[ExternalApp]:
        """Get apps that support the given file type."""
        ext = file_path.suffix.lower()
        return [app for app in self._apps if ext in app.file_types]

    def open_with_default(self, file_path: Path):
        """Open file with system default application."""
        os.startfile(str(file_path))

    def open_with_app(self, file_path: Path, app: ExternalApp):
        """Open file with specified application."""
        args = app.arguments.replace("{file}", str(file_path))
        cmd = f'"{app.path}" {args}'
        subprocess.Popen(cmd, shell=True)


# =============================================================================
# Abstract Metadata Handler (Open/Closed Principle)
# =============================================================================


class MetadataHandler(ABC):
    """Abstract base class for metadata handlers (Dependency Inversion)."""

    @abstractmethod
    def can_handle(self, file_path: Path) -> bool:
        """Check if this handler can process the given file."""
        pass

    @abstractmethod
    def read_metadata(self, file_path: Path) -> BookMetadata:
        """Read metadata from the file."""
        pass

    @abstractmethod
    def write_metadata(self, file_path: Path, metadata: BookMetadata) -> None:
        """Write metadata to the file."""
        pass


# =============================================================================
# Concrete Metadata Handlers
# =============================================================================


class PDFMetadataHandler(MetadataHandler):
    """Handler for PDF metadata operations."""

    def can_handle(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".pdf"

    def read_metadata(self, file_path: Path) -> BookMetadata:
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError(
                "pypdf is required for PDF support. Install with: pip install pypdf"
            )

        reader = PdfReader(str(file_path))
        info = reader.metadata or {}

        return BookMetadata(
            title=info.get("/Title", "") or "",
            author=info.get("/Author", "") or "",
            subject=info.get("/Subject", "") or "",
            keywords=info.get("/Keywords", "") or "",
        )

    def write_metadata(self, file_path: Path, metadata: BookMetadata) -> None:
        try:
            from pypdf import PdfReader, PdfWriter
        except ImportError:
            raise ImportError(
                "pypdf is required for PDF support. Install with: pip install pypdf"
            )

        reader = PdfReader(str(file_path))
        writer = PdfWriter()

        for page in reader.pages:
            writer.add_page(page)

        writer.add_metadata(
            {
                "/Title": metadata.title,
                "/Author": metadata.author,
                "/Subject": metadata.subject,
                "/Keywords": metadata.keywords,
            }
        )

        # Write to temp file first, then replace
        temp_path = file_path.with_suffix(".tmp")
        with open(temp_path, "wb") as f:
            writer.write(f)

        # Replace original with temp
        os.replace(temp_path, file_path)


class EPUBMetadataHandler(MetadataHandler):
    """Handler for EPUB metadata operations."""

    def can_handle(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".epub"

    def read_metadata(self, file_path: Path) -> BookMetadata:
        try:
            from ebooklib import epub
        except ImportError:
            raise ImportError(
                "ebooklib is required for EPUB support. Install with: pip install ebooklib"
            )

        options = {"ignore_ncx": True}
        book = epub.read_epub(str(file_path), options=options)

        title = book.get_metadata("DC", "title")
        author = book.get_metadata("DC", "creator")
        subject = book.get_metadata("DC", "subject")

        return BookMetadata(
            title=title[0][0] if title else "",
            author=author[0][0] if author else "",
            subject="; ".join([s[0] for s in subject]) if subject else "",
            keywords="",
        )

    def write_metadata(self, file_path: Path, metadata: BookMetadata) -> None:
        try:
            from ebooklib import epub
        except ImportError:
            raise ImportError(
                "ebooklib is required for EPUB support. Install with: pip install ebooklib"
            )

        # Read the epub with options to handle various NCX issues
        options = {"ignore_ncx": True}
        book = epub.read_epub(str(file_path), options=options)

        # Clear existing metadata safely
        dc_namespace = "http://purl.org/dc/elements/1.1/"
        if dc_namespace in book.metadata:
            if "title" in book.metadata[dc_namespace]:
                book.metadata[dc_namespace]["title"] = []
            if "creator" in book.metadata[dc_namespace]:
                book.metadata[dc_namespace]["creator"] = []
            if "subject" in book.metadata[dc_namespace]:
                book.metadata[dc_namespace]["subject"] = []

        # Set new metadata
        book.set_title(metadata.title)
        if metadata.author:
            book.add_author(metadata.author)
        if metadata.subject:
            for subj in metadata.subject.split(";"):
                subj = subj.strip()
                if subj:
                    book.add_metadata("DC", "subject", subj)

        # Write to temp file first, then replace (avoids bad zip file error)
        temp_path = file_path.with_suffix(".epub.tmp")

        # Write with options to skip validation that might fail on missing NCX
        epub_options = {"epub3_pages": False, "play_order": False}
        epub.write_epub(str(temp_path), book, epub_options)

        # Replace original with temp
        os.replace(temp_path, file_path)


# =============================================================================
# Handler Registry (Open/Closed Principle)
# =============================================================================


class MetadataHandlerRegistry:
    """Registry for metadata handlers, allowing easy extension."""

    def __init__(self):
        self._handlers: List[MetadataHandler] = []

    def register(self, handler: MetadataHandler) -> None:
        self._handlers.append(handler)

    def get_handler(self, file_path: Path) -> Optional[MetadataHandler]:
        for handler in self._handlers:
            if handler.can_handle(file_path):
                return handler
        return None

    def get_supported_extensions(self) -> List[str]:
        """Get list of supported file extensions."""
        extensions = []
        for handler in self._handlers:
            if isinstance(handler, PDFMetadataHandler):
                extensions.append(".pdf")
            elif isinstance(handler, EPUBMetadataHandler):
                extensions.append(".epub")
        return extensions


# =============================================================================
# File Service (Single Responsibility)
# =============================================================================


class FileService:
    """Service for file operations."""

    def __init__(self, registry: MetadataHandlerRegistry):
        self._registry = registry

    def load_file(self, file_path: Path) -> BookFile:
        """Load a single file and read its metadata."""
        book = BookFile(path=file_path)
        handler = self._registry.get_handler(file_path)

        if not handler:
            book.error = "Unsupported file format"
            return book

        try:
            metadata = handler.read_metadata(file_path)
            book.original_metadata = metadata
            book.current_metadata = metadata.copy()
        except Exception as e:
            book.error = str(e)

        return book

    def save_file(self, book: BookFile) -> None:
        """Save metadata to file."""
        handler = self._registry.get_handler(book.path)
        if not handler:
            raise ValueError("Unsupported file format")

        handler.write_metadata(book.path, book.current_metadata)
        book.original_metadata = book.current_metadata.copy()

    def rename_file(self, book: BookFile, new_name: str) -> Path:
        """Rename file and return new path."""
        new_path = book.path.parent / new_name
        
        # Check if path would be too long (Windows MAX_PATH limit)
        if len(str(new_path)) > 259:
            raise ValueError(
                f"Path too long ({len(str(new_path))} characters). "
                f"Windows limit is 260. Try shortening the title or author."
            )
        
        book.path.rename(new_path)
        book.path = new_path
        return new_path

    def scan_directory(self, directory: Path, recursive: bool = True) -> List[Path]:
        """Scan directory for supported files."""
        extensions = self._registry.get_supported_extensions()
        files = []

        if recursive:
            for ext in extensions:
                files.extend(directory.rglob(f"*{ext}"))
        else:
            for ext in extensions:
                files.extend(directory.glob(f"*{ext}"))

        return sorted(files)


# =============================================================================
# Background Worker (Single Responsibility)
# =============================================================================


class BackgroundWorker:
    """Handles background task execution with callbacks to main thread."""

    def __init__(self, root: tk.Tk):
        self._root = root
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._task_queue = queue.Queue()
        self._running = True
        self._poll_interval = 100  # ms
        self._poll()

    def submit(
        self,
        task: Callable,
        callback: Optional[Callable] = None,
        error_callback: Optional[Callable] = None,
    ) -> None:
        """Submit a task to run in background."""

        def wrapped_task():
            try:
                result = task()
                self._task_queue.put(("success", callback, result))
            except Exception as e:
                self._task_queue.put(("error", error_callback, e))

        self._executor.submit(wrapped_task)

    def _poll(self) -> None:
        """Poll for completed tasks and run callbacks on main thread."""
        while not self._task_queue.empty():
            status, callback, result = self._task_queue.get_nowait()
            if callback:
                callback(result)

        if self._running:
            self._root.after(self._poll_interval, self._poll)

    def shutdown(self) -> None:
        """Shutdown the worker."""
        self._running = False
        self._executor.shutdown(wait=False)


# =============================================================================
# Filename Parser Service
# =============================================================================


class FilenameParser:
    """Service for parsing and generating filenames."""

    SEPARATORS = [" - ", " _ ", " – ", " — ", "_", "-"]

    def parse_to_metadata(
        self, filename: str, author_first: bool = False
    ) -> BookMetadata:
        """Parse filename to extract title and author."""
        # Remove extension
        name = Path(filename).stem

        # Try to find separator
        for sep in self.SEPARATORS:
            if sep in name:
                parts = name.split(sep, 1)
                if len(parts) == 2:
                    if author_first:
                        return BookMetadata(
                            author=parts[0].strip(), title=parts[1].strip()
                        )
                    else:
                        return BookMetadata(
                            title=parts[0].strip(), author=parts[1].strip()
                        )

        # No separator found, use whole name as title
        return BookMetadata(title=name)

    def generate_filename(
        self,
        metadata: BookMetadata,
        extension: str,
        author_first: bool = False,
        separator: str = " - ",
        max_length: int = 200,  # Leave room for path + extension
    ) -> str:
        """Generate filename from metadata."""
        title = self._sanitize_filename(metadata.title) or "Untitled"
        author = self._sanitize_filename(metadata.author)
        
        # Truncate author if it contains multiple authors (usually separated by &, and, or ,)
        if author and len(author) > 50:
            # Keep only first author if multiple
            for sep in [' & ', ', ', ' and ']:
                if sep in author:
                    author = author.split(sep)[0].strip()
                    break
            # Still too long? Truncate
            if len(author) > 50:
                author = author[:47] + "..."

        if author:
            if author_first:
                name = f"{author}{separator}{title}"
            else:
                name = f"{title}{separator}{author}"
        else:
            name = title
        
        # Ensure total filename length doesn't exceed max_length
        max_name_length = max_length - len(extension)
        if len(name) > max_name_length:
            # Calculate how much to keep
            if author:
                # Try to keep both title and author, but truncate title first
                available = max_name_length - len(author) - len(separator) - 3  # 3 for "..."
                if available > 20:  # Minimum reasonable title length
                    title_truncated = title[:available] + "..."
                    if author_first:
                        name = f"{author}{separator}{title_truncated}"
                    else:
                        name = f"{title_truncated}{separator}{author}"
                else:
                    # Not enough room, use title only and truncate
                    name = title[:max_name_length - 3] + "..."
            else:
                # Title only, just truncate
                name = name[:max_name_length - 3] + "..."

        return f"{name}{extension}"

    def _sanitize_filename(self, name: str) -> str:
        """Remove invalid filename characters."""
        if not name:
            return ""
        # Remove invalid characters
        invalid_chars = r'[<>:"/\\|?*]'
        return re.sub(invalid_chars, "", name).strip()


# =============================================================================
# Custom Widgets
# =============================================================================


class ScrollableFrame(ttk.Frame):
    """A scrollable frame widget."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        # Create canvas and scrollbar
        self.canvas = tk.Canvas(self, highlightthickness=0, bg="#f0f0f0")
        self.scrollbar = ttk.Scrollbar(
            self, orient="vertical", command=self.canvas.yview
        )
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>", lambda e: self._update_scroll_region()
        )

        self.canvas_frame = self.canvas.create_window(
            (0, 0), window=self.scrollable_frame, anchor="nw"
        )
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Bind canvas resize to adjust frame width
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # Pack widgets
        self.scrollbar.pack(side="left", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        # Bind mousewheel
        self.bind_mousewheel()

    def _update_scroll_region(self):
        """Update scroll region to match content size."""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfig(self.canvas_frame, width=event.width)
        # Update scroll region when canvas is resized
        self._update_scroll_region()

    def bind_mousewheel(self):
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def unbind_mousewheel(self):
        self.canvas.unbind_all("<MouseWheel>")

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


class FileListItem(ttk.Frame):
    """A single item in the file list."""

    def __init__(
        self,
        parent,
        book: BookFile,
        on_select: Callable,
        on_context_menu: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(parent, **kwargs)
        self.book = book
        self.on_select = on_select
        self.on_context_menu = on_context_menu
        self._selected = False

        # Configure style
        self.configure(padding=3)

        # Create widgets
        self._create_widgets()

        # Bind click
        self.bind("<Button-1>", self._on_click)
        self.bind("<Button-3>", self._on_right_click)
        self._bind_children_click(self)

    def _bind_children_click(self, widget):
        """Recursively bind click to all children."""
        for child in widget.winfo_children():
            child.bind("<Button-1>", self._on_click)
            child.bind("<Button-3>", self._on_right_click)
            self._bind_children_click(child)

    def _on_right_click(self, event):
        """Handle right click for context menu."""
        self.on_select(self)  # Select the item first
        if self.on_context_menu:
            self.on_context_menu(event, self.book)

    def _create_widgets(self):
        # Left side: status and icon
        left_frame = ttk.Frame(self)
        left_frame.pack(side="left", padx=(0, 5))

        # Status indicator
        self.status_label = ttk.Label(left_frame, text="●", width=2)
        self.status_label.pack(side="left")

        # File icon based on type
        icon = "📕" if self.book.extension == ".pdf" else "📗"
        self.icon_label = ttk.Label(left_frame, text=icon, width=2)
        self.icon_label.pack(side="left")

        # Right side: file info (filename + metadata)
        info_frame = ttk.Frame(self)
        info_frame.pack(side="left", fill="x", expand=True)

        # Filename
        self.name_label = ttk.Label(
            info_frame, text=self.book.filename, anchor="w", style="FileItem.TLabel"
        )
        self.name_label.pack(fill="x", anchor="w")

        # Title line
        self.title_label = ttk.Label(
            info_frame, text="", anchor="w", style="FileItemMeta.TLabel"
        )
        self.title_label.pack(fill="x", anchor="w")

        # Author line
        self.author_label = ttk.Label(
            info_frame, text="", anchor="w", style="FileItemMeta.TLabel"
        )
        self.author_label.pack(fill="x", anchor="w")

        # Error line (only shown if error exists)
        self.error_label = ttk.Label(
            info_frame, text="", anchor="w", foreground="red", font=("Segoe UI", 7)
        )

        self.update_status()
        self.update_metadata_display()

    def _on_click(self, event):
        self.on_select(self)

    def set_selected(self, selected: bool):
        self._selected = selected
        self._apply_selection_style(self, selected)

    def _apply_selection_style(self, widget, selected: bool):
        """Recursively apply selection style to widget and children."""
        if isinstance(widget, ttk.Frame):
            widget.configure(style="Selected.TFrame" if selected else "TFrame")
        elif isinstance(widget, ttk.Label):
            # Don't change style for error label (keep red)
            if widget == getattr(self, "error_label", None):
                bg = "#0078d4" if selected else "#f0f0f0"
                widget.configure(background=bg)
            # Check if it's a meta label (title or author)
            elif widget in (
                getattr(self, "title_label", None),
                getattr(self, "author_label", None),
            ):
                widget.configure(
                    style="Selected.FileItemMeta.TLabel"
                    if selected
                    else "FileItemMeta.TLabel"
                )
            else:
                widget.configure(style="Selected.TLabel" if selected else "TLabel")

        for child in widget.winfo_children():
            self._apply_selection_style(child, selected)

    def update_status(self):
        if self.book.error:
            self.status_label.configure(foreground="red")
        elif self.book.is_modified:
            self.status_label.configure(foreground="orange")
        else:
            self.status_label.configure(foreground="green")

    def update_filename(self):
        self.name_label.configure(text=self.book.filename)

    def update_metadata_display(self):
        """Update the metadata display lines."""
        if self.book.error:
            # Show error message instead of metadata
            self.title_label.configure(text="")
            self.author_label.configure(text="")
            self.error_label.configure(text=f"⚠️ Error: {self.book.error}")
            self.error_label.pack(fill="x", anchor="w")
        else:
            # Hide error label and show metadata
            self.error_label.pack_forget()

            title = self.book.current_metadata.title
            author = self.book.current_metadata.author

            # Display title (or placeholder if empty)
            if title:
                self.title_label.configure(text=f"📖 {title}")
            else:
                self.title_label.configure(text="📖 (no title)")

            # Display author (or placeholder if empty)
            if author:
                self.author_label.configure(text=f"✍ {author}")
            else:
                self.author_label.configure(text="✍ (no author)")


# =============================================================================
# Main Application (Single Responsibility for UI)
# =============================================================================


class MetadataEditorApp:
    """Main application class."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Ebook Metadata Editor")
        self.root.geometry("1000x700")
        self.root.minsize(800, 500)

        # Initialize services
        self._init_services()

        # Setup styles
        self._setup_styles()

        # Create UI
        self._create_menu()
        self._create_ui()

        # State
        self.books: Dict[Path, BookFile] = {}
        self.file_items: Dict[Path, FileListItem] = {}
        self.current_book: Optional[BookFile] = None
        self._loading_files = False  # Track when batch loading
        self._pending_files_count = 0  # Track pending loads
    def _init_services(self):
        """Initialize all services."""
        # Create handler registry
        self.registry = MetadataHandlerRegistry()
        self.registry.register(PDFMetadataHandler())
        self.registry.register(EPUBMetadataHandler())

        # Create services
        self.file_service = FileService(self.registry)
        self.filename_parser = FilenameParser()
        self.worker = BackgroundWorker(self.root)
        self.external_app_manager = ExternalAppManager()

    def _setup_styles(self):
        """Setup ttk styles."""
        style = ttk.Style()

        # Try to use a modern theme
        available_themes = style.theme_names()
        if "clam" in available_themes:
            style.theme_use("clam")

        # Custom styles
        style.configure("TFrame", background="#f0f0f0")
        style.configure("TLabel", background="#f0f0f0", font=("Segoe UI", 8))
        style.configure("TButton", font=("Segoe UI", 8), padding=3)
        style.configure("TEntry", font=("Segoe UI", 8), padding=3)

        # Selected item style
        style.configure("Selected.TFrame", background="#0078d4")
        style.configure("Selected.TLabel", background="#0078d4", foreground="white")

        # Header style
        style.configure("Header.TLabel", font=("Segoe UI", 9, "bold"))

        # Status bar style
        style.configure("Status.TLabel", font=("Segoe UI", 8), padding=3)

        # File list item styles
        style.configure("FileItem.TLabel", font=("Segoe UI", 8))
        style.configure(
            "FileItemMeta.TLabel", font=("Segoe UI", 7), foreground="#666666"
        )
        style.configure(
            "Selected.FileItemMeta.TLabel",
            font=("Segoe UI", 7),
            background="#0078d4",
            foreground="#cccccc",
        )

    def _create_menu(self):
        """Create application menu."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(
            label="Open Files...", command=self._open_files, accelerator="Ctrl+O"
        )
        file_menu.add_command(
            label="Open Folder...",
            command=self._open_folder,
            accelerator="Ctrl+Shift+O",
        )
        file_menu.add_separator()
        file_menu.add_command(
            label="Save Current", command=self._save_current, accelerator="Ctrl+S"
        )
        file_menu.add_command(
            label="Save All", command=self._save_all, accelerator="Ctrl+Shift+S"
        )
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_close)

        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(
            label="Import from Filename (Title - Author)",
            command=lambda: self._import_from_filename(False),
        )
        edit_menu.add_command(
            label="Import from Filename (Author - Title)",
            command=lambda: self._import_from_filename(True),
        )
        edit_menu.add_separator()
        edit_menu.add_command(
            label="Rename File from Metadata", command=self._rename_from_metadata
        )
        edit_menu.add_separator()
        edit_menu.add_command(label="Revert Changes", command=self._revert_changes)

        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(
            label="Show Files with Errors", command=self._show_error_files
        )
        view_menu.add_command(
            label="Show Modified Files", command=self._show_modified_files
        )
        view_menu.add_command(label="Show All Files", command=self._show_all_files)

        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(
            label="Batch Import from Filenames...", command=self._batch_import_filenames
        )
        tools_menu.add_command(
            label="Batch Rename from Metadata...", command=self._batch_rename
        )

        # Settings menu
        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Settings", menu=settings_menu)
        settings_menu.add_command(
            label="Configure External Apps...", command=self._configure_external_apps
        )

        # Bind keyboard shortcuts
        self.root.bind("<Control-o>", lambda e: self._open_files())
        self.root.bind("<Control-O>", lambda e: self._open_folder())
        self.root.bind("<Control-s>", lambda e: self._save_current())
        self.root.bind("<Control-S>", lambda e: self._save_all())

    def _create_ui(self):
        """Create main UI."""
        # Main container
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill="both", expand=True)

        # Create paned window for resizable panels
        paned = ttk.PanedWindow(main_frame, orient="horizontal")
        paned.pack(fill="both", expand=True)

        # Left panel - File list
        left_frame = ttk.Frame(paned, width=300)
        self._create_file_list_panel(left_frame)
        paned.add(left_frame, weight=1)

        # Right panel - Editor
        right_frame = ttk.Frame(paned)
        self._create_editor_panel(right_frame)
        paned.add(right_frame, weight=2)

        # Status bar
        self._create_status_bar()

    def _create_file_list_panel(self, parent):
        """Create file list panel."""
        # Header
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill="x", pady=(0, 10))

        ttk.Label(header_frame, text="Files", style="Header.TLabel").pack(side="left")

        # Button frame
        btn_frame = ttk.Frame(header_frame)
        btn_frame.pack(side="right")

        ttk.Button(btn_frame, text="📁", width=3, command=self._open_files).pack(
            side="left", padx=2
        )
        ttk.Button(btn_frame, text="📂", width=3, command=self._open_folder).pack(
            side="left", padx=2
        )
        ttk.Button(btn_frame, text="🗑", width=3, command=self._remove_selected).pack(
            side="left", padx=2
        )

        # Scrollable file list
        self.file_list_frame = ScrollableFrame(parent)
        self.file_list_frame.pack(fill="both", expand=True)

        # Legend
        legend_frame = ttk.Frame(parent)
        legend_frame.pack(fill="x", pady=(10, 0))

        ttk.Label(legend_frame, text="●", foreground="green").pack(side="left")
        ttk.Label(legend_frame, text="Saved").pack(side="left", padx=(0, 10))
        ttk.Label(legend_frame, text="●", foreground="orange").pack(side="left")
        ttk.Label(legend_frame, text="Modified").pack(side="left", padx=(0, 10))
        ttk.Label(legend_frame, text="●", foreground="red").pack(side="left")
        ttk.Label(legend_frame, text="Error").pack(side="left")

    def _create_editor_panel(self, parent):
        """Create metadata editor panel."""
        # Header
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill="x", pady=(0, 10))

        ttk.Label(header_frame, text="Metadata Editor", style="Header.TLabel").pack(
            side="left"
        )

        # Current file label
        self.current_file_label = ttk.Label(header_frame, text="No file selected")
        self.current_file_label.pack(side="right")

        # Editor form
        form_frame = ttk.LabelFrame(parent, text="Metadata", padding=15)
        form_frame.pack(fill="both", expand=True, pady=(0, 10))

        # Configure grid
        form_frame.columnconfigure(1, weight=1)

        # Title
        ttk.Label(form_frame, text="Title:").grid(row=0, column=0, sticky="w", pady=3)
        self.title_var = tk.StringVar()
        self.title_entry = ttk.Entry(
            form_frame, textvariable=self.title_var, font=("Segoe UI", 9)
        )
        self.title_entry.grid(row=0, column=1, sticky="ew", pady=3, padx=(10, 0))
        self.title_var.trace_add("write", self._on_metadata_change)

        # Author
        ttk.Label(form_frame, text="Author:").grid(row=1, column=0, sticky="w", pady=3)
        self.author_var = tk.StringVar()
        self.author_entry = ttk.Entry(
            form_frame, textvariable=self.author_var, font=("Segoe UI", 9)
        )
        self.author_entry.grid(row=1, column=1, sticky="ew", pady=3, padx=(10, 0))
        self.author_var.trace_add("write", self._on_metadata_change)

        # Subject
        ttk.Label(form_frame, text="Subject:").grid(row=2, column=0, sticky="w", pady=3)
        self.subject_var = tk.StringVar()
        self.subject_entry = ttk.Entry(
            form_frame, textvariable=self.subject_var, font=("Segoe UI", 9)
        )
        self.subject_entry.grid(row=2, column=1, sticky="ew", pady=3, padx=(10, 0))
        self.subject_var.trace_add("write", self._on_metadata_change)

        # Keywords
        ttk.Label(form_frame, text="Keywords:").grid(
            row=3, column=0, sticky="w", pady=3
        )
        self.keywords_var = tk.StringVar()
        self.keywords_entry = ttk.Entry(
            form_frame, textvariable=self.keywords_var, font=("Segoe UI", 9)
        )
        self.keywords_entry.grid(row=3, column=1, sticky="ew", pady=3, padx=(10, 0))
        self.keywords_var.trace_add("write", self._on_metadata_change)

        # Quick actions
        actions_frame = ttk.LabelFrame(parent, text="Quick Actions", padding=15)
        actions_frame.pack(fill="x", pady=(0, 10))

        btn_row1 = ttk.Frame(actions_frame)
        btn_row1.pack(fill="x", pady=(0, 5))

        ttk.Button(
            btn_row1,
            text="📥 Import from Filename (Title - Author)",
            command=lambda: self._import_from_filename(False),
        ).pack(side="left", padx=(0, 5))
        ttk.Button(
            btn_row1,
            text="📥 Import from Filename (Author - Title)",
            command=lambda: self._import_from_filename(True),
        ).pack(side="left")

        btn_row2 = ttk.Frame(actions_frame)
        btn_row2.pack(fill="x")

        ttk.Button(
            btn_row2,
            text="📝 Rename File from Metadata",
            command=self._rename_from_metadata,
        ).pack(side="left", padx=(0, 5))
        ttk.Button(
            btn_row2, text="↩ Revert Changes", command=self._revert_changes
        ).pack(side="left")

        # Save buttons
        save_frame = ttk.Frame(parent)
        save_frame.pack(fill="x")

        ttk.Button(save_frame, text="💾 Save Current", command=self._save_current).pack(
            side="left", padx=(0, 5)
        )
        ttk.Button(
            save_frame, text="💾 Save All Modified", command=self._save_all
        ).pack(side="left")

    def _create_status_bar(self):
        """Create status bar."""
        self.status_bar = ttk.Label(
            self.root, text="Ready", style="Status.TLabel", anchor="w"
        )
        self.status_bar.pack(fill="x", side="bottom")

    def _set_status(self, message: str):
        """Update status bar message."""
        self.status_bar.configure(text=message)

    # =========================================================================
    # File Operations
    # =========================================================================

    def _open_files(self):
        """Open file dialog to select files."""
        filetypes = [
            ("Ebook files", "*.pdf *.epub"),
            ("PDF files", "*.pdf"),
            ("EPUB files", "*.epub"),
            ("All files", "*.*"),
        ]

        files = filedialog.askopenfilenames(
            title="Select Ebook Files", filetypes=filetypes
        )

        if files:
            self._load_files([Path(f) for f in files])

    def _open_folder(self):
        """Open folder dialog to scan for files."""
        folder = filedialog.askdirectory(title="Select Folder to Scan")

        if folder:
            self._set_status(f"Scanning folder: {folder}")

            def scan():
                return self.file_service.scan_directory(Path(folder), recursive=True)

            def on_complete(files):
                self._load_files(files)

            self.worker.submit(scan, on_complete, lambda e: self._show_error(str(e)))

    def _load_files(self, paths: List[Path]):
        """Load multiple files."""
        # Filter out already loaded files
        new_paths = [p for p in paths if p not in self.books]
        
        if not new_paths:
            return
        
        self._set_status(f"Loading {len(new_paths)} files...")
        self._loading_files = True
        self._pending_files_count = len(new_paths)

        for path in new_paths:
            def load(p=path):
                return self.file_service.load_file(p)

            def on_complete(book, p=path):
                self.books[p] = book
                self._add_file_item(book)
                self._pending_files_count -= 1
                
                # Only sort when all files are loaded
                if self._pending_files_count == 0:
                    self._loading_files = False
                    self._sort_file_list()
                    self._update_file_count()
                    self._set_status(f"Loaded {len(new_paths)} files")

            self.worker.submit(
                load, on_complete, lambda e: self._show_error(str(e))
            )

    def _add_file_item(self, book: BookFile):
        """Add file item to the list."""
        item = FileListItem(
            self.file_list_frame.scrollable_frame,
            book,
            self._on_file_select,
            self._show_file_context_menu,
        )
        # Don't pack yet, will be sorted
        self.file_items[book.path] = item
        
        # Only sort immediately if not batch loading
        if not self._loading_files:
            self._sort_file_list()

    def _sort_file_list(self):
        """Sort file list with errors on top, then by filename."""
        # Get all books sorted: errors first, then by filename
        sorted_books = sorted(
            self.books.values(),
            key=lambda b: (
                not b.error,
                b.filename.lower(),
            ),  # False (error) comes before True (no error)
        )

        # Repack items in sorted order
        for book in sorted_books:
            if book.path in self.file_items:
                item = self.file_items[book.path]
                item.pack_forget()
                item.pack(fill="x", pady=1)

        # Update scroll region
        self.file_list_frame.canvas.update_idletasks()
        self.file_list_frame._update_scroll_region()

    def _on_file_select(self, item: FileListItem):
        """Handle file selection."""
        # Deselect previous
        if self.current_book and self.current_book.path in self.file_items:
            self.file_items[self.current_book.path].set_selected(False)

        # Select new
        item.set_selected(True)
        self.current_book = item.book

        # Update editor
        self._load_editor(item.book)

    def _load_editor(self, book: BookFile):
        """Load book metadata into editor."""
        self.current_file_label.configure(text=book.filename)

        # Temporarily disable trace to avoid triggering change detection
        self._updating_editor = True

        self.title_var.set(book.current_metadata.title)
        self.author_var.set(book.current_metadata.author)
        self.subject_var.set(book.current_metadata.subject)
        self.keywords_var.set(book.current_metadata.keywords)

        self._updating_editor = False

        if book.error:
            self._set_status(f"Error: {book.error}")
        else:
            self._set_status(f"Loaded: {book.filename}")

    def _on_metadata_change(self, *args):
        """Handle metadata change in editor."""
        if hasattr(self, "_updating_editor") and self._updating_editor:
            return

        if not self.current_book:
            return

        # Update current book metadata
        self.current_book.current_metadata.title = self.title_var.get()
        self.current_book.current_metadata.author = self.author_var.get()
        self.current_book.current_metadata.subject = self.subject_var.get()
        self.current_book.current_metadata.keywords = self.keywords_var.get()

        # Update status indicator and metadata display in file list
        if self.current_book.path in self.file_items:
            self.file_items[self.current_book.path].update_status()
            self.file_items[self.current_book.path].update_metadata_display()

    def _remove_selected(self):
        """Remove selected file from list."""
        if not self.current_book:
            return

        path = self.current_book.path

        # Remove from books
        del self.books[path]

        # Remove from UI
        item = self.file_items.pop(path)
        item.destroy()

        # Clear selection
        self.current_book = None
        self._clear_editor()
        self._update_file_count()

    def _clear_editor(self):
        """Clear editor fields."""
        self._updating_editor = True
        self.title_var.set("")
        self.author_var.set("")
        self.subject_var.set("")
        self.keywords_var.set("")
        self.current_file_label.configure(text="No file selected")
        self._updating_editor = False

    def _update_file_count(self):
        """Update status with file count."""
        total = len(self.books)
        modified = sum(1 for b in self.books.values() if b.is_modified)
        errors = sum(1 for b in self.books.values() if b.error)

        if errors > 0:
            self._set_status(
                f"Files: {total} | Modified: {modified} | Errors: {errors}"
            )
        else:
            self._set_status(f"Files: {total} | Modified: {modified}")

    # =========================================================================
    # Save Operations
    # =========================================================================

    def _save_current(self):
        """Save current file."""
        if not self.current_book:
            return

        if not self.current_book.is_modified:
            self._set_status("No changes to save")
            return

        self._set_status(f"Saving: {self.current_book.filename}")

        book = self.current_book

        def save():
            self.file_service.save_file(book)

        def on_complete(_):
            if book.path in self.file_items:
                self.file_items[book.path].update_status()
            self._set_status(f"Saved: {book.filename}")
            self._update_file_count()

        self.worker.submit(
            save, on_complete, lambda e: self._show_error(f"Save failed: {e}")
        )

    def _save_all(self):
        """Save all modified files."""
        modified = [b for b in self.books.values() if b.is_modified]

        if not modified:
            self._set_status("No files to save")
            return

        self._set_status(f"Saving {len(modified)} files...")

        saved_count = [0]
        total = len(modified)

        for book in modified:

            def save(b=book):
                self.file_service.save_file(b)
                return b

            def on_complete(b):
                saved_count[0] += 1
                if b.path in self.file_items:
                    self.file_items[b.path].update_status()
                if saved_count[0] == total:
                    self._set_status(f"Saved {total} files")
                    self._update_file_count()
                    # Resort in case error status changed
                    self._sort_file_list()

            self.worker.submit(
                save, on_complete, lambda e: self._show_error(f"Save failed: {e}")
            )

    # =========================================================================
    # Import/Rename Operations
    # =========================================================================

    def _import_from_filename(self, author_first: bool):
        """Import metadata from filename."""
        if not self.current_book:
            return

        parsed = self.filename_parser.parse_to_metadata(
            self.current_book.filename, author_first
        )

        if parsed.title:
            self.title_var.set(parsed.title)
        if parsed.author:
            self.author_var.set(parsed.author)

        self._set_status("Imported metadata from filename")

    def _rename_from_metadata(self):
        """Rename file based on metadata."""
        if not self.current_book:
            return
        
        # Calculate max filename length based on directory path
        dir_path_length = len(str(self.current_book.path.parent))
        # Windows MAX_PATH is 260, leave some margin and account for directory separator
        max_filename_length = 255 - dir_path_length - 1  # -1 for path separator
        max_filename_length = min(max_filename_length, 200)  # Cap at 200 for safety

        new_name = self.filename_parser.generate_filename(
            self.current_book.current_metadata, 
            self.current_book.extension,
            max_length=max_filename_length
        )

        if new_name == self.current_book.filename:
            self._set_status("Filename unchanged")
            return

        # Confirm rename
        if not messagebox.askyesno("Rename File", f"Rename file to:\n{new_name}"):
            return

        book = self.current_book
        item = self.file_items[book.path]
        old_path = book.path

        def rename():
            return self.file_service.rename_file(book, new_name)

        def on_complete(new_path):
            # Update dictionaries with new path
            del self.books[old_path]
            del self.file_items[old_path]
            self.books[new_path] = book
            self.file_items[new_path] = item

            # Update UI
            item.update_filename()
            self.current_file_label.configure(text=book.filename)
            self._set_status(f"Renamed to: {new_name}")

        self.worker.submit(
            rename, on_complete, lambda e: self._show_error(f"Rename failed: {e}")
        )

    def _revert_changes(self):
        """Revert changes to current file."""
        if not self.current_book:
            return

        self.current_book.current_metadata = self.current_book.original_metadata.copy()
        self._load_editor(self.current_book)

        if self.current_book.path in self.file_items:
            self.file_items[self.current_book.path].update_status()

        self._set_status("Changes reverted")

    # =========================================================================
    # Batch Operations
    # =========================================================================

    def _batch_import_filenames(self):
        """Batch import metadata from filenames."""
        if not self.books:
            messagebox.showinfo("Info", "No files loaded")
            return

        # Create dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Batch Import from Filenames")
        dialog.geometry("400x200")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(
            dialog, text="Import metadata from filenames for all files", padding=20
        ).pack()

        format_var = tk.StringVar(value="title_author")

        ttk.Radiobutton(
            dialog,
            text="Title - Author format",
            variable=format_var,
            value="title_author",
        ).pack(pady=5)
        ttk.Radiobutton(
            dialog,
            text="Author - Title format",
            variable=format_var,
            value="author_title",
        ).pack(pady=5)

        def apply():
            author_first = format_var.get() == "author_title"
            count = 0

            for book in self.books.values():
                if not book.error:
                    parsed = self.filename_parser.parse_to_metadata(
                        book.filename, author_first
                    )
                    if parsed.title:
                        book.current_metadata.title = parsed.title
                    if parsed.author:
                        book.current_metadata.author = parsed.author

                    if book.path in self.file_items:
                        self.file_items[book.path].update_status()
                    count += 1

            # Refresh current editor
            if self.current_book:
                self._load_editor(self.current_book)

            dialog.destroy()
            self._set_status(f"Imported metadata for {count} files")
            self._update_file_count()

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=20)
        ttk.Button(btn_frame, text="Apply", command=apply).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(
            side="left", padx=5
        )

    def _batch_rename(self):
        """Batch rename files from metadata."""
        modified = [b for b in self.books.values() if not b.error]

        if not modified:
            messagebox.showinfo("Info", "No files to rename")
            return

        # Create dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Batch Rename from Metadata")
        dialog.geometry("400x250")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text="Rename files based on metadata", padding=20).pack()

        format_var = tk.StringVar(value="title_author")

        ttk.Radiobutton(
            dialog,
            text="Title - Author format",
            variable=format_var,
            value="title_author",
        ).pack(pady=5)
        ttk.Radiobutton(
            dialog,
            text="Author - Title format",
            variable=format_var,
            value="author_title",
        ).pack(pady=5)

        sep_var = tk.StringVar(value=" - ")
        sep_frame = ttk.Frame(dialog)
        sep_frame.pack(pady=10)
        ttk.Label(sep_frame, text="Separator:").pack(side="left")
        sep_entry = ttk.Entry(sep_frame, textvariable=sep_var, width=10)
        sep_entry.pack(side="left", padx=5)

        def apply():
            author_first = format_var.get() == "author_title"
            separator = sep_var.get()
            renamed_count = [0]
            total = len(modified)

            for book in modified:
                old_path = book.path
                item = self.file_items.get(old_path)
                
                # Calculate max filename length based on directory path
                dir_path_length = len(str(book.path.parent))
                max_filename_length = 255 - dir_path_length - 1
                max_filename_length = min(max_filename_length, 200)

                new_name = self.filename_parser.generate_filename(
                    book.current_metadata, 
                    book.extension, 
                    author_first, 
                    separator,
                    max_filename_length
                )

                if new_name == book.filename:
                    renamed_count[0] += 1
                    continue

                def rename(b=book, n=new_name, op=old_path, it=item):
                    return (self.file_service.rename_file(b, n), b, op, it)

                def on_complete(result):
                    new_path, b, op, it = result
                    # Update dictionaries
                    if op in self.books:
                        del self.books[op]
                    if op in self.file_items:
                        del self.file_items[op]
                    self.books[new_path] = b
                    if it:
                        self.file_items[new_path] = it
                        it.update_filename()

                    renamed_count[0] += 1
                    if renamed_count[0] == total:
                        if self.current_book:
                            self.current_file_label.configure(
                                text=self.current_book.filename
                            )
                        self._set_status(f"Renamed {total} files")

                self.worker.submit(
                    rename,
                    on_complete,
                    lambda e: self._show_error(f"Rename failed: {e}"),
                )

            dialog.destroy()

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=20)
        ttk.Button(btn_frame, text="Apply", command=apply).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(
            side="left", padx=5
        )

    # =========================================================================
    # Context Menu and External Apps
    # =========================================================================

    def _show_file_context_menu(self, event, book: BookFile):
        """Show context menu for file item."""
        menu = tk.Menu(self.root, tearoff=0)

        # Open with default app
        menu.add_command(
            label="📂 Open with Default App",
            command=lambda: self._open_with_default(book),
        )

        # Open file location
        menu.add_command(
            label="📁 Open File Location",
            command=lambda: self._open_file_location(book),
        )

        # Get configured apps for this file type
        apps = self.external_app_manager.get_apps_for_file(book.path)

        if apps:
            menu.add_separator()
            menu.add_command(label="Open with:", state="disabled")

            for app in apps:
                menu.add_command(
                    label=f"    {app.name}",
                    command=lambda a=app: self._open_with_app(book, a),
                )

        menu.add_separator()
        menu.add_command(
            label="⚙ Configure External Apps...", command=self._configure_external_apps
        )

        # Show menu
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def _open_with_default(self, book: BookFile):
        """Open file with system default application."""
        try:
            self.external_app_manager.open_with_default(book.path)
            self._set_status(f"Opened: {book.filename}")
        except Exception as e:
            self._show_error(f"Failed to open file: {e}")

    def _open_file_location(self, book: BookFile):
        """Open the folder containing the file."""
        try:
            # On Windows, select the file in Explorer
            subprocess.run(["explorer", "/select,", str(book.path)])
            self._set_status(f"Opened folder: {book.path.parent}")
        except Exception as e:
            self._show_error(f"Failed to open folder: {e}")

    def _open_with_app(self, book: BookFile, app: ExternalApp):
        """Open file with specified application."""
        try:
            self.external_app_manager.open_with_app(book.path, app)
            self._set_status(f"Opened {book.filename} with {app.name}")
        except Exception as e:
            self._show_error(f"Failed to open with {app.name}: {e}")

    def _configure_external_apps(self):
        """Open dialog to configure external applications."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Configure External Apps")
        dialog.geometry("600x450")
        dialog.transient(self.root)
        dialog.grab_set()

        # Instructions
        ttk.Label(
            dialog,
            text="Configure external applications to open your ebooks.\nUse {file} in arguments as placeholder for the file path.",
            padding=10,
        ).pack(fill="x")

        # Main frame with list and buttons
        main_frame = ttk.Frame(dialog, padding=10)
        main_frame.pack(fill="both", expand=True)

        # App list with scrollbar
        list_frame = ttk.Frame(main_frame)
        list_frame.pack(side="left", fill="both", expand=True)

        # Treeview for apps
        columns = ("name", "path", "types")
        tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=10)
        tree.heading("name", text="Name")
        tree.heading("path", text="Path")
        tree.heading("types", text="File Types")
        tree.column("name", width=120)
        tree.column("path", width=280)
        tree.column("types", width=80)

        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)

        tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def refresh_list():
            tree.delete(*tree.get_children())
            for i, app in enumerate(self.external_app_manager.apps):
                tree.insert(
                    "",
                    "end",
                    iid=str(i),
                    values=(app.name, app.path, ", ".join(app.file_types)),
                )

        refresh_list()

        # Buttons frame
        btn_frame = ttk.Frame(main_frame, padding=(10, 0, 0, 0))
        btn_frame.pack(side="right", fill="y")

        def add_app():
            self._edit_external_app_dialog(
                dialog,
                None,
                lambda app: (self.external_app_manager.add_app(app), refresh_list()),
            )

        def edit_app():
            selection = tree.selection()
            if not selection:
                return
            index = int(selection[0])
            app = self.external_app_manager.apps[index]
            self._edit_external_app_dialog(
                dialog,
                app,
                lambda new_app: (
                    self.external_app_manager.update_app(index, new_app),
                    refresh_list(),
                ),
            )

        def remove_app():
            selection = tree.selection()
            if not selection:
                return
            index = int(selection[0])
            if messagebox.askyesno("Confirm", "Remove this application?"):
                self.external_app_manager.remove_app(index)
                refresh_list()

        def move_up():
            selection = tree.selection()
            if not selection:
                return
            index = int(selection[0])
            if index > 0:
                self.external_app_manager.move_app(index, -1)
                refresh_list()
                tree.selection_set(str(index - 1))

        def move_down():
            selection = tree.selection()
            if not selection:
                return
            index = int(selection[0])
            if index < len(self.external_app_manager.apps) - 1:
                self.external_app_manager.move_app(index, 1)
                refresh_list()
                tree.selection_set(str(index + 1))

        ttk.Button(btn_frame, text="Add...", command=add_app, width=12).pack(pady=2)
        ttk.Button(btn_frame, text="Edit...", command=edit_app, width=12).pack(pady=2)
        ttk.Button(btn_frame, text="Remove", command=remove_app, width=12).pack(pady=2)
        ttk.Separator(btn_frame, orient="horizontal").pack(fill="x", pady=10)
        ttk.Button(btn_frame, text="Move Up", command=move_up, width=12).pack(pady=2)
        ttk.Button(btn_frame, text="Move Down", command=move_down, width=12).pack(
            pady=2
        )

        # Close button
        ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=10)

    def _edit_external_app_dialog(
        self, parent, app: Optional[ExternalApp], on_save: Callable
    ):
        """Dialog to add or edit an external application."""
        dialog = tk.Toplevel(parent)
        dialog.title("Edit External App" if app else "Add External App")
        dialog.geometry("500x300")
        dialog.transient(parent)
        dialog.grab_set()

        form_frame = ttk.Frame(dialog, padding=20)
        form_frame.pack(fill="both", expand=True)
        form_frame.columnconfigure(1, weight=1)

        # Name
        ttk.Label(form_frame, text="Name:").grid(row=0, column=0, sticky="w", pady=5)
        name_var = tk.StringVar(value=app.name if app else "")
        name_entry = ttk.Entry(form_frame, textvariable=name_var)
        name_entry.grid(row=0, column=1, columnspan=2, sticky="ew", pady=5)

        # Path
        ttk.Label(form_frame, text="Path:").grid(row=1, column=0, sticky="w", pady=5)
        path_var = tk.StringVar(value=app.path if app else "")
        path_entry = ttk.Entry(form_frame, textvariable=path_var)
        path_entry.grid(row=1, column=1, sticky="ew", pady=5)

        def browse_path():
            file_path = filedialog.askopenfilename(
                title="Select Application",
                filetypes=[("Executable files", "*.exe"), ("All files", "*.*")],
            )
            if file_path:
                path_var.set(file_path)

        ttk.Button(form_frame, text="Browse...", command=browse_path).grid(
            row=1, column=2, pady=5, padx=(5, 0)
        )

        # Arguments
        ttk.Label(form_frame, text="Arguments:").grid(
            row=2, column=0, sticky="w", pady=5
        )
        args_var = tk.StringVar(value=app.arguments if app else "{file}")
        args_entry = ttk.Entry(form_frame, textvariable=args_var)
        args_entry.grid(row=2, column=1, columnspan=2, sticky="ew", pady=5)
        ttk.Label(
            form_frame,
            text="Use {file} as placeholder for file path",
            font=("Segoe UI", 7),
        ).grid(row=3, column=1, sticky="w")

        # File types
        ttk.Label(form_frame, text="File Types:").grid(
            row=4, column=0, sticky="w", pady=(15, 5)
        )

        types_frame = ttk.Frame(form_frame)
        types_frame.grid(row=4, column=1, columnspan=2, sticky="w", pady=(15, 5))

        pdf_var = tk.BooleanVar(
            value=app is None or ".pdf" in app.file_types if app else True
        )
        epub_var = tk.BooleanVar(
            value=app is None or ".epub" in app.file_types if app else True
        )

        ttk.Checkbutton(types_frame, text="PDF", variable=pdf_var).pack(
            side="left", padx=(0, 10)
        )
        ttk.Checkbutton(types_frame, text="EPUB", variable=epub_var).pack(side="left")

        def save():
            name = name_var.get().strip()
            path = path_var.get().strip()

            if not name:
                messagebox.showerror("Error", "Name is required", parent=dialog)
                return
            if not path:
                messagebox.showerror("Error", "Path is required", parent=dialog)
                return

            file_types = []
            if pdf_var.get():
                file_types.append(".pdf")
            if epub_var.get():
                file_types.append(".epub")

            if not file_types:
                messagebox.showerror(
                    "Error", "Select at least one file type", parent=dialog
                )
                return

            new_app = ExternalApp(
                name=name, path=path, arguments=args_var.get(), file_types=file_types
            )
            on_save(new_app)
            dialog.destroy()

        # Buttons
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="Save", command=save).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(
            side="left", padx=5
        )

    # =========================================================================
    # File Filtering
    # =========================================================================

    def _show_error_files(self):
        """Show only files with errors."""
        count = 0
        for path, item in self.file_items.items():
            book = self.books[path]
            if book.error:
                item.pack(fill="x", pady=1)
                count += 1
            else:
                item.pack_forget()

        if count == 0:
            messagebox.showinfo("No Errors", "No files with errors found!")
            self._show_all_files()
        else:
            self._set_status(f"Showing {count} file(s) with errors")

    def _show_modified_files(self):
        """Show only modified files."""
        count = 0
        for path, item in self.file_items.items():
            book = self.books[path]
            if book.is_modified and not book.error:
                item.pack(fill="x", pady=1)
                count += 1
            else:
                item.pack_forget()

        if count == 0:
            messagebox.showinfo("No Modified Files", "No modified files found!")
            self._show_all_files()
        else:
            self._set_status(f"Showing {count} modified file(s)")

    def _show_all_files(self):
        """Show all files."""
        for item in self.file_items.values():
            item.pack(fill="x", pady=1)
        self._update_file_count()

    # =========================================================================
    # Utilities
    # =========================================================================

    def _show_error(self, message: str):
        """Show error message."""
        self._set_status(f"Error: {message}")
        messagebox.showerror("Error", message)

    def _on_close(self):
        """Handle window close."""
        # Check for unsaved changes
        modified = [b for b in self.books.values() if b.is_modified]

        if modified:
            result = messagebox.askyesnocancel(
                "Unsaved Changes",
                f"You have {len(modified)} unsaved file(s). Save before closing?",
            )

            if result is None:  # Cancel
                return
            elif result:  # Yes - save
                self._save_all()

        self.worker.shutdown()
        self.root.destroy()


# =============================================================================
# Entry Point
# =============================================================================


def main():
    root = tk.Tk()
    app = MetadataEditorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
