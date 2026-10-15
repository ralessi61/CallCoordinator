# CallsCoordinator 
iPython package that provides a simple and efficient way to coordinate and manage calls in a distributed system. It allows developers to define workflows, handle retries, and manage dependencies between different services.
### Installation
```powershell
poetry install
```
after, virtual environment is ready
### discover tests
#### Anything after poetry run uses that environment, so pydantic and the project package are available
```powershell
poetry run python -m unittest discover -s tests -t .
```
### Tests
```powershell
poetry run python -c "import calls_coordinator"
poetry run pytest
poetry run calls-coordinator
```
