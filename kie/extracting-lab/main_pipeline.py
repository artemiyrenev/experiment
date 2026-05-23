import os
import subprocess
import re
from bs4 import BeautifulSoup


# БЛОК 1: ОБРАБОТКА EXCEL -> HTML
def convert_excel_to_html(input_file, output_dir):
    """ Конвертирует Excel (с его микростолбцами 1С) в резиновый HTML """
    print(f"🔄 Конвертирую в HTML: {os.path.basename(input_file)}...")
    try:
        subprocess.run(
            ['libreoffice', '--headless', '--convert-to', 'html', '--outdir', output_dir, input_file],
            check=True, capture_output=True
        )
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        result_path = os.path.join(output_dir, f"{base_name}.html")
        return result_path if os.path.exists(result_path) else None
    except Exception as e:
        print(f"❌ Ошибка LibreOffice: {e}")
        return None


# БЛОК 2: ОБРАБОТКА HTML -> TEXT
def parse_html(html_path, output_file):
    """ Достает данные из HTML: удаляет мусорные пробелы и дедуплицирует шапки страниц """
    print(f"📊 Извлекаю данные из HTML: {os.path.basename(html_path)}")
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
            
        seen_lines = set() # Хранилище для глобальной дедупликации
            
        with open(output_file, 'w', encoding='utf-8') as out:
            out.write("[ ДАННЫЕ ДОКУМЕНТА (EXCEL ORIGIN) ]\n")
            
            for row in soup.find_all('tr'):
                cells = []
                for td in row.find_all(['td', 'th']):
                    text = td.get_text(separator=" ", strip=True)
                    if text:
                        text = re.sub(r' +', ' ', text)
                        text = re.sub(r' ,', ',', text)
                        cells.append(text)
                    else:
                        cells.append("")
                
                if any(cells):
                    line = "|".join(cells)
                    
                    # 2. Удаляем конструкции |--|--|-- в самом конце строки
                    line = re.sub(r'(?:\|--)+$', '', line)
                    
                    # 1. Удаляем все лишние палки (|||||) в конце строки
                    line = line.rstrip('|')
                    
                    # 3. Если в строке есть подстрока из более чем 10 палок подряд - меняем её на пробел
                    line = re.sub(r'\|{11,}', ' ', line)

                    # Если после очистки строка стала пустой, пропускаем её
                    if not line.strip():
                        continue

                    if line in seen_lines:
                        continue
                        
                    seen_lines.add(line)
                    out.write(line + "\n")
                    
        return True
    except Exception as e:
        print(f"❌ Ошибка парсинга HTML: {e}")
        return False


# БЛОК 3: PDF -> TEXT
def process_pdf_to_text(pdf_path, output_file):
    """ Использует мощный C++ движок Poppler для идеального сохранения layout'а 1С """
    print(f"📄 Poppler (pdftotext) анализирует: {os.path.basename(pdf_path)}")
    
    # Сначала сохраняем во временный файл
    tmp_out = output_file + ".tmp"
    
    try:
        subprocess.run(
            ['pdftotext', '-layout', '-nopgbrk', pdf_path, tmp_out],
            check=True
        )
        
        # Читаем результат и чистим от бесконечных пустых строк
        with open(tmp_out, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("[ ДАННЫЕ ДОКУМЕНТА (PDF ORIGIN) ]\n\n")
            for line in lines:
                # Оставляем только строки, в которых есть хоть какой-то текст
                if line.strip():
                    f.write(line)
                    
        return True
        
    except FileNotFoundError:
        print("❌ ОШИБКА: Не установлен Poppler! Выполни в консоли: sudo apt install poppler-utils")
        return False
    except Exception as e:
        print(f"❌ Ошибка pdftotext: {e}")
        return False
    finally:
        if os.path.exists(tmp_out):
            os.remove(tmp_out)

# БЛОК 4: ОРКЕСТРАТОР ПАЙПЛАЙНА
def process_dataset(base_dir):
    if not os.path.exists(base_dir):
        print(f"❌ Директория {base_dir} не найдена!")
        return

    for folder_name in sorted(os.listdir(base_dir)):
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.isdir(folder_path): 
            print("не найдено")
            continue

        print(f"\n📂 Обработка папки: {folder_name}")
        
        target_doc = None
        doc_ext = None
        
        for file in os.listdir(folder_path):
            if file.startswith('.~lock'): 
                continue
            ext = file.lower().split('.')[-1]
            if ext in ['pdf', 'xls', 'xlsx']:
                target_doc = os.path.join(folder_path, file)
                doc_ext = ext
                break 

        if not target_doc: 
            print("⚠️ Исходные файлы не найдены. Пропуск.")
            continue
            
        prompt_output_path = os.path.join(folder_path, "prompt.txt")
        files_to_cleanup = []

        try:
            if doc_ext == 'pdf':
                process_pdf_to_text(target_doc, prompt_output_path)
                
            elif doc_ext in ['xls', 'xlsx']:
                temp_html = convert_excel_to_html(target_doc, folder_path)
                if temp_html:
                    files_to_cleanup.append(temp_html)
                    parse_html(temp_html, prompt_output_path)
                    
        finally:
            for tmp_file in files_to_cleanup:
                if os.path.exists(tmp_file): 
                    os.remove(tmp_file)
            
            if os.path.exists(prompt_output_path):
                print(f"✅ Успешно сохранен контекст для LLM: prompt.txt")
            else:
                print(f"❌ Не удалось создать prompt.txt")

if __name__ == "__main__":
    DATASET_PATH = "../../example_dataset"
    process_dataset(DATASET_PATH)
    print("Обработка датасета завершена")