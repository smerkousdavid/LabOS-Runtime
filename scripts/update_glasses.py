#!/usr/bin/env python3
"""Glasses configuration utility.

Detects USB-connected VITURE XR glasses and writes the runtime's IP/port
so the glasses app knows where to connect.

Can be run standalone or called from launcher.py.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from rich.console import Console
from rich.table import Table

from glass_connectors import USBConnector
from network_utils import get_network_interfaces

console = Console()

DEFAULT_PORT = 5050


def select_network_interface() -> tuple[str, int]:
    """Interactive network interface selection. Returns (ip, port)."""
    interfaces = get_network_interfaces()

    if not interfaces:
        console.print("[red]No active network interfaces found.[/red]")
        sys.exit(1)

    table = Table(title="Available Network Interfaces")
    table.add_column("#", style="bold")
    table.add_column("Interface")
    table.add_column("IP Address", style="cyan")
    table.add_column("Type")

    for i, (name, ip, kind) in enumerate(interfaces, 1):
        table.add_row(str(i), name, ip, kind)

    console.print(table)
    console.print()

    while True:
        choice = console.input(f"Select interface [1-{len(interfaces)}]: ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(interfaces):
                break
        except ValueError:
            pass
        console.print("[yellow]Invalid selection, try again.[/yellow]")

    _, selected_ip, _ = interfaces[idx]

    port_input = console.input(f"gRPC port [default {DEFAULT_PORT}]: ").strip()
    port = int(port_input) if port_input else DEFAULT_PORT

    return selected_ip, port


def run_update():
    """Main update flow."""
    console.print()
    console.print("[bold]LabOS Glasses Configuration Utility[/bold]")
    console.print()
    console.print("This will write the runtime connection config to your VITURE glasses.")
    console.print()

    selected_ip, port = select_network_interface()
    console.print(f"\nSelected: [cyan]{selected_ip}:{port}[/cyan]")
    console.print()

    console.print("Connect your VITURE glasses via USB cable and make sure they are powered on.")
    console.print("Waiting for device to mount...", style="dim")

    connector = USBConnector()
    mount = connector.wait_for_device(timeout=120, poll_interval=2.0)

    if mount is None:
        console.print("[red]Timed out waiting for glasses. Check the USB connection.[/red]")
        sys.exit(1)

    console.print(f"[green]Device detected at:[/green] {mount}")

    existing = connector.read_config(mount)
    if existing:
        console.print(f"  Current config: {existing}")

    success = connector.write_config(selected_ip, port, mount)
    if not success:
        console.print("[red]Failed to write config to glasses.[/red]")
        sys.exit(1)

    console.print()
    console.print(f"[green bold]Glasses configured to connect to {selected_ip}:{port}[/green bold]")
    console.print()
    console.print("Next steps:")
    console.print("  1. Make sure glasses and this computer are on the same network")
    console.print("  2. Unplug the glasses")
    console.print("  3. Power cycle the glasses (turn off, then on)")
    console.print("  4. Start the runtime: ./run.sh or run.bat")
    console.print()


if __name__ == "__main__":
    run_update()
