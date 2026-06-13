# Template Tesis UDINUS

Template LaTeX untuk tesis Program Magister Teknik Informatika, Universitas Dian Nuswantoro.

## Struktur Folder

```
thesis-template/
├── thesis.tex              # File utama
├── settings/               # Konfigurasi LaTeX
│   ├── packages.tex        # Daftar paket
│   ├── metadata.tex        # Metadata (judul, nama, pembimbing)
│   ├── commands.tex         # Custom commands
│   ├── formatting.tex      # Format halaman, heading, dll
│   └── numbering.tex       # Penomoran list
├── frontmatter/            # Halaman depan
│   ├── titlepage.tex       # Halaman sampul
│   ├── title_inner.tex     # Halaman judul
│   ├── approval_status.tex # Pengesahan status tesis
│   ├── statement.tex       # Pernyataan penulis
│   ├── approval.tex        # Persetujuan tesis
│   ├── approval_pengesahan.tex  # Pengesahan tesis
│   ├── abstract_en.tex     # Abstract (English)
│   ├── abstract_id.tex     # Abstrak (Indonesia)
│   ├── acknowledgements.tex    # Ucapan terima kasih
│   ├── daftar_publikasi.tex    # Daftar publikasi
│   ├── list_of_appendices.tex  # Daftar lampiran
│   └── list_of_terms.tex   # Daftar istilah
├── chapters/               # Bab-bab tesis
│   ├── bab1.tex            # Pendahuluan
│   ├── bab2.tex            # Tinjauan pustaka
│   ├── bab3.tex            # Metodologi
│   ├── bab4.tex            # Hasil dan pembahasan
│   └── bab5.tex            # Kesimpulan dan saran
├── references/             # Referensi
│   ├── references.tex      # Bibliography style
│   └── references.bib      # File BibTeX
├── appendices/             # Lampiran
│   ├── turnitin.tex        # Contoh lampiran turnitin
│   └── loa_publikasi.tex   # Contoh lampiran LOA publikasi
└── assets/                 # Aset (gambar, logo)
    ├── logos/
    │   └── logo_udinus.png
    └── figures/
        └── common/
```

## Instalasi

### Prasyarat

- **TeX Distribution**: Distribusi LaTeX (MacTeX, TeX Live, atau MiKTeX)
- **Editor**: VS Code + extension LaTeX Workshop (rekomendasi) atau TeXstudio, TeXmaker
- **Font**: Times New Roman, Arial, Courier New (biasanya sudah ada di sistem)

---

## macOS

### Opsi 1: MacTeX (Rekomendasi)

1. **Download MacTeX**
   ```bash
   # Menggunakan Homebrew
   brew install --cask mactex
   
   # Atau download manual dari:
   # https://www.tug.org/mactex/mactex-download.html
   ```

2. **Update PATH** (restart terminal setelah instalasi)
   ```bash
   # Tambahkan ke shell config (zsh)
   echo 'export PATH="/Library/TeX/texbin:$PATH"' >> ~/.zshrc
   source ~/.zshrc
   
   # Atau bash
   echo 'export PATH="/Library/TeX/texbin:$PATH"' >> ~/.bash_profile
   source ~/.bash_profile
   ```

3. **Verifikasi instalasi**
   ```bash
   pdflatex --version
   latexmk --version
   ```

### Opsi 2: TeX Live via Homebrew (Minimal)

```bash
brew install texlive
```

---

## Windows

### Opsi 1: MiKTeX (Rekomendasi untuk Windows)

1. **Download MiKTeX**
   - Kunjungi: https://miktex.org/download
   - Download installer untuk Windows (64-bit atau 32-bit)
   - Jalankan installer, pilih "Install for all users"

2. **Konfigurasi**
   - Pilih "Automatic package installation" saat instalasi
   - Tambahkan ke PATH otomatis (cek opsi installer)

3. **Verifikasi instalasi**
   ```cmd
   pdflatex --version
   latexmk --version
   ```

### Opsi 2: TeX Live (Alternatif)

1. **Download TeX Live**
   - Kunjungi: https://www.tug.org/texlive/acquire-netinstall.html
   - Download `install-tl-windows.exe`

2. **Instalasi**
   - Jalankan installer
   - Pilih "Full install" (recommended)
   - Lokasi default: `C:\texlive\2024`

3. **Verifikasi instalasi**
   ```cmd
   pdflatex --version
   ```

---

## Linux

### Debian/Ubuntu

```bash
# Update package list
sudo apt update

# Install TeX Live full
sudo apt install texlive-full

# Untuk instalasi minimal
sudo apt install texlive-latex-base texlive-fonts-recommended texlive-latex-extra

# Install latexmk dan tools tambahan
sudo apt install latexmk biber

# Install development tools (opsional)
sudo apt install texlive-bibtex-extra texlive-lang-other
```

### Fedora

```bash
# Install TeX Live
sudo dnf install texlive-scheme-full

# Atau minimal
sudo dnf install texlive-latex texlive-latexmk
```

### Arch Linux

```bash
# Install TeX Live
sudo pacman -S texlive-most

# Atau minimal
sudo pacman -S texlive-core texlive-latexextra

# Install latexmk
sudo pacman -S latexmk
```

### Verifikasi instalasi

```bash
pdflatex --version
latexmk --version
```

---

## VS Code Extensions

### Wajib (Required)

#### 1. LaTeX Workshop

- **Extension ID**: `James-Yu.latex-workshop`
- **Fungsi**: Compile, preview, syntax highlighting, autoc omplete
- **Install**:
  - Buka VS Code
  - Tekan `Cmd/Ctrl + Shift + X`
  - Cari "LaTeX Workshop"
  - Klik Install

#### 2. LaTeX Language Support

- **Extension ID**: `tecosaur.latex-tools`
- **Fungsi**: Improved syntax highlighting, snippets lanjutan

### Rekomendasi (Optional)

#### 3. BibTeX Extension

- **Extension ID**: `mblode.twig-language-2`

#### 4. Code Spell Checker

- **Extension ID**: `streetsidesoftware.code-spell-checker`
- **Fungsi**: Cek ejaan bahasa Inggris
- **Install additional dictionaries**:
  ```
  @idaeal resource "@idaeal/dictionaries"
  ```

#### 5. Word Counter

- **Extension ID**: `ms-vscode.wordcount`
- **Fungsi**: Hitung kata (berguna untuk tracking progress)

#### 6. GitLens (untuk version control)

- **Extension ID**: `eamodio.gitlens`
- **Fungsi**: Git integration untuk tracking changes

### Konfigurasi VS Code

Buat file `.vscode/settings.json` di folder project:

```json
{
    "latex-workshop.latex.recipes": [
        {
            "name": "pdfLaTeX + BibTeX",
            "tools": ["pdflatex", "bibtex", "pdflatex", "pdflatex"]
        },
        {
            "name": "latexmk",
            "tools": ["latexmk"]
        }
    ],
    "latex-workshop.latex.tools": [
        {
            "name": "pdflatex",
            "command": "pdflatex",
            "args": ["-synctex=1", "-interaction=nonstopmode", "-shell-escape", "%DOC%"]
        },
        {
            "name": "bibtex",
            "command": "bibtex",
            "args": ["%DOCFILE%"]
        },
        {
            "name": "latexmk",
            "command": "latexmk",
            "args": ["-pdf", "-synctex=1", "-interaction=nonstopmode", "-shell-escape", "%DOC%"]
        }
    ],
    "latex-workshop.view.pdf.viewer": "tab",
    "latex-workshop.latex.autoBuild.run": "onSave",
    "latex-workshop.latex.clean.fileTypes": [
        "*.aux",
        "*.bbl",
        "*.blg",
        "*.idx",
        "*.ind",
        "*.lof",
        "*.lot",
        "*.out",
        "*.toc",
        "*.acn",
        "*.acr",
        "*.alg",
        "*.glg",
        "*.glo",
        "*.gls",
        "*.fls",
        "*.log",
        "*.fdb_latexmk",
        "*.synctex.gz"
    ]
}
```

## Cara Build

### Metode 1: LaTeX Workshop (VS Code)

1. Buka file `thesis.tex` di VS Code
2. Tekan `Cmd/Ctrl + S` untuk save (auto-build) atau `Cmd/Ctrl + Option/Alt + B`
3. Preview PDF: `Cmd/Ctrl + Option/Alt + V`

### Metode 2: Terminal (Command Line)

```bash
# Masuk ke folder thesis-template
cd thesis-template

# Build dengan latexmk (rekomendasi)
latexmk -pdf thesis.tex

# Atau build manual (pdflatex + bibtex)
pdflatex thesis.tex
bibtex thesis
pdflatex thesis.tex
pdflatex thesis.tex

# Clean auxiliary files
latexmk -c
```

### Metode 3: Makefile (Opsional)

Buat file `Makefile`:

```makefile
.PHONY: all clean view

all:
	latexmk -pdf thesis.tex

clean:
	latexmk -c
	rm -f *.bbl *.run.xml

view:
	open thesis.pdf  # macOS
	# xdg-open thesis.pdf  # Linux
	# start thesis.pdf     # Windows
```

Gunakan:
```bash
make all      # Build PDF
make clean    # Clean auxiliary files
make view     # Open PDF
```

## Troubleshooting

### Font Times New Roman tidak ditemukan

**macOS/Linux**: Times New Roman biasanya sudah ada. Jika tidak:
```bash
# Linux - install Microsoft fonts
sudo apt install ttf-mscorefonts-installer

# Atau gunakan TeX Gyre Termora sebagai alternatif
# Di packages.tex, ganti Times New Roman dengan:
\setmainfont{TeX Gyre Termora}
```

**Windows**: Times New Roman sudah built-in.

### Package tidak ditemukan

```bash
# TeX Live - install package manual
tlmgr install <package-name>

# MiKTeX - otomatis download saat compile
# Atau: MiKTeX Console -> Settings -> Always missing packages: auto-install
```

### Error "pdflatex command not found"

Pastikan PATH sudah dikonfigurasi:
```bash
# macOS - restart terminal atau:
source ~/.zshrc

# Linux - tambahkan ke .bashrc:
export PATH="/usr/local/texlive/2024/bin/x86_64-linux:$PATH"

# Windows - pastikan MiKTeX/TeX Live ditambahkan ke PATH saat instalasi
```

### PDF tidak update setelah compile

```bash
# Clean dan rebuild
latexmk -c
latexmk -pdf thesis.tex
```

## Customisasi

### Mengubah Metadata

Edit file `settings/metadata.tex`:

```latex
% Ubah judul
\newcommand{\TitleID}{Judul Tesis Anda dalam Bahasa Indonesia}
\newcommand{\TitleEN}{Your Thesis Title in English}

% Ubah identitas
\newcommand{\AuthorName}{Nama Lengkap Mahasiswa}
\newcommand{\StudentID}{NPM123456789}

% Ubah pembimbing
\newcommand{\SupervisorA}{Dr. Nama Pembimbing Utama, M.Kom.}
\newcommand{\SupervisorB}{Nama Pembimbing Pembantu, S.Si, M.Kom}
```

### Menambah Gambar

1. Letakkan file gambar di `assets/figures/<nama-folder>/`
2. Di file `.tex`, tambahkan:

```latex
\begin{figure}[h]
    \centering
    \includegraphics[width=0.8\textwidth]{assets/figures/nama-folder/gambar.png}
    \caption{Deskripsi gambar}
    \label{fig:label_gambar}
\end{figure}
```

3. Referensi dengan `\ref{fig:label_gambar}`

### Menambah Referensi

Edit file `references/references.bib`:

```bibtex
@article{author2024,
    author    = {Author, Name},
    title     = {Article Title},
    journal   = {Journal Name},
    year      = {2024},
    volume    = {1},
    number    = {1},
    pages     = {1--10},
}
```

Di file `.tex`, gunakan `\cite{author2024}`.

## Lisensi

Template ini dibuat untuk keperluan akademik di Universitas Dian Nuswantoro.

## Bantuan

Untuk bantuan lebih lanjut, hubungi:
- Email: [email departemen]
- Website: https://fti.dinus.ac.id