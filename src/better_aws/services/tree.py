from dataclasses import dataclass, field
from typing import Dict, List, Optional
from rich.console import Console
from rich.tree import Tree
from rich.text import Text

def _human_bytes(n: int) -> str:
    """
    Converts a byte count into a human-readable format using binary prefixes.

    Parameters
    ----------
    n : int
        The number of bytes to convert.
    
    Returns
    -------
    str
        A human-readable string representing the byte count, using appropriate units (B, KiB, MiB, etc.).
    """
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    x = float(n or 0)
    for u in units:
        if x < 1024.0 or u == units[-1]:
            if u == "B":
                return f"{int(x)} {u}"
            return f"{x:.2f} {u}"
        x /= 1024.0
    return f"{x:.2f} PiB"

@dataclass
class _Node:
    """
    Represents a node in the tree structure of S3 objects.
    Each node can be either a file or a directory.

    Attributes
    ----------
    name : str
        The name of the node (file or directory).
    full_path : str
        The full path of the node from the root.
    is_file : bool, optional
        Indicates whether the node is a file (True) or a directory (False). Defaults to False.
    size : int, optional
        The size of the file in bytes. For directories, this is typically 0. Defaults to 0.
    children : Dict[str, "_Node"], optional
        A dictionary mapping child node names to their corresponding _Node instances. Defaults to an empty dictionary.
    """
    name: str
    full_path: str
    is_file: bool = False
    size: int = 0
    children: Dict[str, "_Node"] = field(default_factory=dict)

def _build_tree_from_objects(objects: List[dict], root_label: str = "") -> _Node:
    """
    Builds a tree structure from a list of S3 objects.

    Parameters
    ----------
    objects : List[dict]
        A list of S3 objects, where each object is a dictionary containing at least a "key" and optionally a "size".
    root_label : str, optional
        The label for the root node of the tree. Defaults to an empty string.

    Returns
    -------
    _Node
        The root node of the constructed tree.
    """
    root = _Node(name=root_label or "/", full_path=root_label or "/")
    for o in objects:
        key = o["key"]
        size = int(o.get("size", 0) or 0)
        parts = [p for p in key.split("/") if p]
        cur = root
        acc_path = ""
        for i, part in enumerate(parts):
            acc_path = f"{acc_path}/{part}" if acc_path else part
            is_file = (i == len(parts) - 1)
            if part not in cur.children:
                cur.children[part] = _Node(
                    name=part,
                    full_path=acc_path,
                    is_file=is_file,
                    size=0,
                )
            cur = cur.children[part]
            if is_file:
                cur.is_file = True
                cur.size = size
    return root

def _compute_folder_sizes(node: _Node) -> int:
    """
    Recursively computes the total size of each folder in the tree.
    For files, it returns their size.

    Parameters
    ----------
    node : _Node
        The node for which to compute the size.
    
    Returns
    -------
    int
        The total size of the node.
    """
    if node.is_file:
        return node.size
    total = 0
    for child in node.children.values():
        total += _compute_folder_sizes(child)
    node.size = total
    return total

def _sorted_children(node: _Node, folders_first: bool = True) -> List[_Node]:
    """
    Returns the children of a node sorted by size and name.

    Parameters
    ----------
    node : _Node
        The node whose children are to be sorted.
    folders_first : bool, optional
        If True, folders will be listed before files. Defaults to True.
    
    Returns
    -------
    List[_Node]
        A list of child nodes sorted by size and name.
    """
    kids = list(node.children.values())
    if folders_first:
        return sorted(
            kids,
            key=lambda n: (0 if not n.is_file else 1, -n.size, n.name.lower()),
        )
    return sorted(kids, key=lambda n: (-n.size, n.name.lower()))

def _render_tree(
    root: _Node,
    *,
    show_full_path: bool = True,
    max_depth: Optional[int] = None,
    max_children: Optional[int] = None,
    folders_first: bool = True,
) -> Tree:
    """
    Renders the tree structure as a rich Tree object.

    Parameters
    ----------
    root : _Node
        The root node of the tree to render.
    show_full_path : bool, optional
        If True, the full path of each node will be shown. If False, only the
        basename will be shown. Defaults to True.
    max_depth : Optional[int], optional
        The maximum depth to display. If None, there is no limit. Defaults to None.
    max_children : Optional[int], optional
        The maximum number of children to display for each node. If None, there is no limit. Defaults to None.
    folders_first : bool, optional
        If True, folders will be listed before files. Defaults to True.

    Returns
    -------
    Tree
        A rich Tree object representing the structure of the S3 objects.
    """
    root_text = Text(root.full_path if show_full_path else root.name)
    root_text.append(f"  ({_human_bytes(root.size)})", style="dim")
    t = Tree(root_text)

    def add(node: _Node, tree: Tree, depth: int) -> None:
        if max_depth is not None and depth >= max_depth:
            return

        kids = _sorted_children(node, folders_first=folders_first)

        if max_children is not None and len(kids) > max_children:
            shown = kids[:max_children]
            hidden = kids[max_children:]
        else:
            shown = kids
            hidden = []

        for c in shown:
            if show_full_path:
                parent = c.full_path.rsplit("/", 1)[0] if "/" in c.full_path else ""
                base = c.name
                line = Text()
                if parent:
                    line.append(parent + "/", style="dim")
                line.append(base, style="bold")
            else:
                line = Text(c.name, style="bold")

            line.append(f"  ({_human_bytes(c.size)})", style="dim")

            child_tree = tree.add(line)
            if not c.is_file:
                add(c, child_tree, depth + 1)

        if hidden:
            hidden_size = sum(x.size for x in hidden)
            more = Text(f"+{len(hidden)} more", style="dim italic")
            more.append(f"  ({_human_bytes(hidden_size)})", style="dim")
            tree.add(more)

    add(root, t, 0)
    return t