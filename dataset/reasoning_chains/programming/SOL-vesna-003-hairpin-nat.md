# Reasoning chain — Hairpin NAT a mental model síťových rolí

## Zdroj
`vesna/SOLUTIONS.md#SOL-003` — `tailscale up --login-server=https://hekate.lomsky.net`
zamrzne uvnitř LXC kontejneru, který běží na stejné LAN jako Headscale
server. Login proces se nikdy nedokončí.

## Kontext
Homelab setup: Hekate (mašina v LAN, IP `192.168.1.20`) hostí Headscale
(self-hosted Tailscale control server). DNS A záznam `hekate.lomsky.net`
ukazuje na **veřejnou IP** routeru (cloudflare DNS). Externí klienti
(z internetu) se tak připojí přes router NAT → Hekate. LXC kontejner
běží také na Hekate (nebo na jiném host v LAN), chce se připojit
k Tailscale s `--login-server=https://hekate.lomsky.net`. **Z LAN klienta
se DNS resolvne na public IP routeru, klient pošle paket na public IP,
router by ho měl forwardovat zpět dovnitř LAN (na Hekate).** Tomu se
říká *hairpin NAT*. TP-Link router to nepodporuje. Paket vyjde na
WAN interface, router zjistí "to je moje IP", a buď ho dropne nebo
loop-em. Connection visí.

## Analytický flow

1. **Pozoruji: Tailscale login zamrzne, žádný error.** Žádný `connection
   refused`, žádný `timeout` (zatím), prostě nic. To je první signal,
   že paket někam jde, ale odpověď nepřichází. Buď se ztratil paket,
   nebo se ztratila response. Žádné DNS error znamená, že DNS
   resolution funguje.

2. **Vykresli si síťovou cestu.** LXC kontejner → bridge (LAN) → Hekate
   network stack → Hekate router (`192.168.1.1`, TP-Link) → ???.
   DNS říká `hekate.lomsky.net` → public IP routeru `89.X.X.X`. Klient
   pošle paket na `89.X.X.X:443`. Paket dorazí na router přes vnitřní
   LAN interface. Router se podívá na destination IP. **Public IP routeru
   = on. Co dělá?** Tady je place, kde různé routery dělají různé věci.

3. **Hairpin NAT: koncept.** Pokud LAN klient pošle paket na public IP
   svého vlastního routeru, ideální chování je: router rozpozná "to
   je moje veřejná IP, port forward by to měl poslat na Hekate
   (192.168.1.20:443)", aplikuje NAT, paket se vrátí dovnitř na
   Hekate. **Tahle operace se jmenuje hairpin NAT (taky NAT loopback).**
   Ne všechny routery to umějí. Levné consumer routery (TP-Link,
   některé D-Link, Asus dle modelu) hairpin nepodporují — paket
   prostě dropnou.

4. **Test hypothesis: ping public IP z LAN.** `ping 89.X.X.X` z LXC
   nebo z jiného LAN klienta. **Nedostává odpověď.** Pingem ven (na
   `8.8.8.8`) odpovědi přicházejí. Confirmed: router nemá hairpin
   NAT, paket s destination = own public IP nedostává response.

5. **Tři cesty k řešení.**
   - **Upgrade router** na model s hairpin NAT support (OpenWrt,
     Mikrotik, EdgeRouter). Velký zásah, hardware investice.
   - **Lokální DNS resolver**, který pro `hekate.lomsky.net` z LAN
     vrací LAN IP (192.168.1.20), z internetu public IP. Tj.
     "split-horizon DNS". Vyžaduje DNS server na Hekate (nebo
     na routeru, pokud rozumí).
   - **Rychlý hack: `/etc/hosts`** uvnitř kontejneru. Override DNS
     resolution pro tento jeden hostname. Žádný extra software,
     pět vteřin práce, funkčí.

6. **Volím (3) — `/etc/hosts` override.** Pro každý LXC kontejner,
   který chce kontaktovat Headscale, přidám:
   ```
   192.168.1.20 hekate.lomsky.net
   ```
   Tailscale teď resolvuje hostname lokálně, paket jde přímo přes
   bridge, žádný hairpin loop. Connection se navazuje.

7. **Reflexe na trade-off.** `/etc/hosts` je trvalý technický dluh —
   každý nový LXC potřebuje stejný workaround, je to jeden řádek
   navíc v provisioning skriptech. Pokud jeden den koupím lepší
   router, můžu odstranit. Mezitím funguje, je traceable (`/etc/hosts`
   je classic, každý sysadmin si to přečte), žádné magické dependencies.

8. **Sekundární uvažování: split-horizon DNS jako better long-term.**
   Pokud bych měl dnsmasq nebo unbound na Hekate (nebo na routeru),
   mohl bych nastavit, že `hekate.lomsky.net` se resolvuje na
   `192.168.1.20` z LAN klientů. Žádný workaround per kontejner,
   centralizovaná správa. Zápis do roadmap, nepriority.

## Aplikovatelné principy

- **Connection "zamrzlá bez error" je signature ztracených paketů.**
  Pokud aplikace/protokol čeká na response, response nepřijde,
  protocol nemá timeout (nebo timeout je dlouhý), uživatel vidí
  "visí". Když problém je ztracený paket, log nekřičí — to je
  silent failure v networking layer.
- **Public IP z LAN klienta vyžaduje hairpin NAT.** Není to
  univerzální feature. Levné routery to nedělají. Před tím, než
  se na to spolehnu (např. self-hosted services accessed both
  externally and internally), vždy testuji `ping public_ip` z LAN.
- **`/etc/hosts` je legitimní debugging i workaround tool.** Není to
  hack v negativním smyslu — je to dokumentovaná, traceable, lokální
  override DNS. Pro one-off či per-host situaci je to ideální. Pro
  více než ~10 hostů přechod na lokální DNS resolver.
- **Síťová cesta je produkt mnoha protocols, každý se vlastní logikou.**
  DNS resolves → IP routing → NAT → firewall → service. Když něco
  nefunguje, identifikuj, *na kterém kroku* se to ztrácí. Bez tohoto
  mental modelu je síťové debugování guess-and-check.

## Závěr

```bash
# /etc/hosts uvnitř každého LXC s Tailscale
echo "192.168.1.20 hekate.lomsky.net" >> /etc/hosts

# Test
ping hekate.lomsky.net  # odpoví z 192.168.1.20, ne z public IP
tailscale up --login-server=https://hekate.lomsky.net  # nyní funguje
```

Long-term roadmap (KI-NNN): nastavit dnsmasq na Hekate jako split-horizon
DNS, který pro queries z 192.168.0.0/16 vrátí `192.168.1.20` pro
`*.lomsky.net`, jinak public IP.

## Přenositelný pattern

Kdykoli debug síťové connectivity a nemám viditelný error:

1. **Vykresli si full network path** — DNS, routing, NAT, firewall,
   protocol handshake. Každý krok je potential failure point.
2. **Test každý krok separately.** `nslookup` pro DNS. `ping` pro
   IP reachability. `traceroute` pro routing. `nc -v port` pro TCP
   connect. `curl -v` pro HTTPS handshake. Každý nástroj testuje
   jiný layer.
3. **Self-loop scénáře (LAN klient → public IP svého routeru) jsou
   special case.** Často nefungují bez hairpin NAT. Cheap test:
   `ping <vlastní public IP>` z LAN klienta. Nepřijde odpověď =
   hairpin nefunguje.
4. **Cheap workaround před expensive infrastruktura change.**
   `/etc/hosts` před lokálním DNS server. Static route před nový
   router. Bridge před VLAN reconfiguration. Workaround dokumentovaný,
   traceable, oddělený do per-host config je legitimní tactic.
5. **Síťové bugs jsou často konfigurační, ne softwarové.** Aplikace
   funguje (kód je OK), ale prostředí ji nepouští. Než hledáš bug
   v aplikaci, ověř že networking layer dělá, co bys čekal.

Pattern se přenáší: VPN setup (split tunneling, route conflicts),
Docker networking (bridge vs. host vs. overlay), Kubernetes services
(DNS, kube-proxy, network policies), inter-service communication
(service mesh, sidecars). Společný invariant: **aplikační logika a
síťová cesta jsou orthogonální dimenze. Bug v jedné se projeví
podobně jako bug v druhé. Disciplína v izolaci layerů (test síť bez
aplikace, otestuj aplikaci s loopback) je rozdíl mezi rychlým fixem
a dvoudenním slepým hledáním.**
