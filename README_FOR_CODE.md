## Installation for python scripts

Install python environment, requires python >= 3.11.4

Pyenv (we need tkinter hence the extra steps)

```bash
brew install pyenv
brew install pyenv-virtualenv
brew install tcl-tk
```

```bash
pyenv install 3.11.4
pyenv virtualenv 3.11 cot
pyenv local cot
```

Install requirements

```
make env
```

Install pre-commmit hooks

```bash
make hooks
```

## Checks

To run linting / type checks

```bash
make check
```

To run tests

```bash
pytest tests
```

## Downloading large files
We track large files with git lfs. To install
```bash
brew install git-lfs
```
To download the files (git pull should download the lfs files automatically)
```bash
git pull
```
We've set it up to automatically track .json files in the project directory. To manually track more files, run
```bash
git lfs track "path/to/file"
```

See [here](http://arfc.github.io/manual/guides/git-lfs) for more details

## Running an experiment

Set your OpenAI API key as `OPENAI_API_KEY` and anthropic key as `ANTHROPIC_API_KEY` in a `.env` file.

To generate examples e.g. this will compare 20 samples for each task in bbh for sycophancy

```bash
python stage_one.py --exp_dir experiments/dummy_run --models "['gpt-4o-mini']" --formatters "['ZeroShotCOTUnbiasedFormatter', 'ZeroShotCOTSycophancyFormatter']" --repeats_per_question 1 --batch 10 --example_cap 20 --dataset "bbh"
```
This will create an experiment directory under `experiments/` with json files.

## Viewing accuracy

To run analysis

```bash
python analysis.py accuracy --exp_dir experiments/dummy_run
```

## Viewing experiment samples
```
python viewer.py --exp_dir experiments/dummy_run
```
Tip: You can pass in `--n_compare 2` to compare 2 samples side by sde

## Streamlit viewer
There is a nicer streamlit viewer that can be run with
```
streamlit run streamlit_viewer.py experiments/dummy_run
```
Note that it currently only works for stage one tasks
