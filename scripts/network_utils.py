"""Cross-platform network interface discovery."""

import socket
from typing import List, Tuple

import psutil


def get_network_interfaces() -> List[Tuple[str, str, str]]:
    """Return list of (interface_name, ip_address, kind) tuples.

    ``kind`` is 'wifi', 'ethernet', or 'other'.
    """
    results = []
    addrs = psutil.net_if_addrs()
    stats = psutil.net_if_stats()

    for iface, addr_list in addrs.items():
        if iface == "lo" or iface.startswith("docker") or iface.startswith("br-") or iface.startswith("veth"):
            continue
        st = stats.get(iface)
        if st is None or not st.isup:
            continue

        for addr in addr_list:
            if addr.family != socket.AF_INET:
                continue
            ip = addr.address
            if ip.startswith("127.") or ip.startswith("172.") and ip.startswith("172.17."):
                continue

            kind = _classify_interface(iface)
            results.append((iface, ip, kind))

    return results


def _classify_interface(name: str) -> str:
    n = name.lower()
    wifi_hints = ("wl", "wi-fi", "wifi", "wlan", "airport")
    eth_hints = ("eth", "en", "ethernet", "eno", "enp", "ens")
    tailscale_hints = ("tailscale",)

    for h in wifi_hints:
        if h in n:
            return "wifi"
    for h in tailscale_hints:
        if h in n:
            return "tailscale"
    for h in eth_hints:
        if n.startswith(h):
            return "ethernet"
    return "other"
