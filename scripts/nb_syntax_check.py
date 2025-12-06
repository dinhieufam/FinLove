import nbformat, pathlib
nb_path = pathlib.Path(r"c:/Users/jimmy/Documents/Tài liệu/vital docs/CODING/Project/FinLove/notebooks/financial_data_eda.ipynb")
nb = nbformat.read(nb_path, as_version=4)
errors = []
for i, cell in enumerate(nb.cells, start=1):
    if cell.cell_type == 'code':
        source = ''.join(cell.source)
        try:
            compile(source, f'<cell {i}>', 'exec')
        except Exception as e:
            errors.append((i, str(e)))
print('Found', len(errors), 'syntax errors')
for i, err in errors:
    print(f'Cell {i}: {err}')
