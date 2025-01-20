<a id="readme-top"></a>

<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]

<!-- MARKDOWN LINKS & IMAGES -->

[contributors-shield]: https://img.shields.io/github/contributors/GHOST-Science-Club/tree-classification-irim.svg?style=for-the-badge

[contributors-url]: https://github.com/GHOST-Science-Club/tree-classification-irim/graphs/contributors

[forks-shield]: https://img.shields.io/github/forks/GHOST-Science-Club/tree-classification-irim.svg?style=for-the-badge

[forks-url]: https://github.com/GHOST-Science-Club/tree-classification-irim/network/members

[stars-shield]: https://img.shields.io/github/stars/GHOST-Science-Club/tree-classification-irim.svg?style=for-the-badge

[stars-url]: https://github.com/GHOST-Science-Club/tree-classification-irim/stargazers

[issues-shield]: https://img.shields.io/github/issues/GHOST-Science-Club/tree-classification-irim.svg?style=for-the-badge

[issues-url]: https://github.com/GHOST-Science-Club/tree-classification-irim/issues

<br />
<a id="readme-top"></a>
<div style="text-align: center;">
  <img src="docs/pexels-markusspiske-1133380.jpeg" alt="Tree classification - Foto from Markus Spiske: https://www.pexels.com/de-de/foto/vogelperspektive-natur-wald-baume-113338/">

<h2 style="text-align: center;">Tree-classification-irim</h2>

  <p style="text-align: center;">
    Repository for the GHOST x IRIM project on tree classification (2024/2025)
    <br />
    <!-- Change link to: https://github.com/GHOST-Science-Club/tree-classification-irim -->
    <a href="https://github.com/GHOST-Science-Club/tree-classification-irim/tree/docu"><b>Explore the docs »</b></a>
    <br />
    <br />
    <!-- What to write? What sections to add?
    <a href="https://github.com/GHOST-Science-Club/tree-classification-irim/tree/docu">View Demo</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues/new?labels=bug&template=bug-report---.md">Report Bug</a> -->
    · 
    <a href="https://github.com/GHOST-Science-Club/tree-classification-irim/issues/new?assignees=&labels=&projects=&template=feature_request.md&title=">Request Feature</a>
    ·
  </p>
</div>

## Table of Contents

- [About the Project](#about-the-project)
- [How to run](#how-to-run)
- [Built With](#built-with)
- [Contributing](#contributing)
- [Communication](#communication)
- [The Team](#the-team)
- [Acknowledgements](#acknowledgments)
- [License](#license)

## About the Project

**GHOST x IRIM** is an initiative by the [GHOST](https://ghost.put.poznan.pl) student organization to develop and test
an AI algorithm for identifying tree species from aerial photos, focusing on Polish forests.

### How to run

Clone the repo and cd into:

```
git clone git@github.com:GHOST-Science-Club/tree-classification-irim.git
cd tree-classification-irim
```

For Linux/MacOS/Unix:

```
python -m venv venv
source venv/bin/activate
pip install -r unix-requirements.txt
python src/main.py
```

For CUDA PyTorch installation, please refer to this link: https://pytorch.org/get-started/locally/ and select your
platform and CUDA.
For Windows with CUDA and win32 lib:

```
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/main.py
```

If you are a project maintainer you should have received Wandb api key.
To send logs from local builds to wandb use:

```
pip install wandb
wandb login
```

Provide your API key when prompted.

### Objectives:

- Identify tree species sensitive to fires, storms, disease, and pests.
- Promote diverse, resilient tree stands over vulnerable monocultures.
- Enable cost-effective, large-scale forest health monitoring.

### Long-Term Goals:

- Present findings at major conferences ([MLinPL](https://mlinpl.org), [GHOST Day](https://ghostday.pl)).
- Contribute to the open-source [Deepness](https://github.com/PUTvision/qgis-plugin-deepness) project.

Supported by the **Institute of Robotics and Machine Intelligence (IRIM)** at **Poznań University of Technology (PUT)**.


<p style="text-align: right;">(<a href="#readme-top">back to top</a>)</p>

## Built With

**Dependencies:**

- **[PyTorch](https://pytorch.org/):** For deep learning model implementation and processing (`torch`, `torchvision`).
- **[PyTorch Lightning](https://www.pytorchlightning.ai/):** Simplifies training workflows (`pytorch-lightning`).
- **[NumPy](https://numpy.org/):** Efficient numerical computing (`numpy`).
- **[Matplotlib](https://matplotlib.org/):** Visualization library for plotting (`matplotlib`).

**Datasets:**

- **[PureForest](https://huggingface.co/datasets/IGNF/PureForest):** Aerial images for tree species classification.

<p style="text-align: right;">(<a href="#readme-top">back to top</a>)</p>

<!-- The specifics of the contribution process to be determined -->

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any
contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also
simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`):
    - the name of the branch should identify the issue it addresses (e.g. issue number and/or some explanation).
    - branch can also be created directly inside an issue: https://shorturl.at/3KkQP.
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`):
    - commit messages should be created according
      to [the Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) framework.
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p style="text-align: right;">(<a href="#readme-top">back to top</a>)</p>

## Communication

- **[GHOST Website](https://ghost.put.poznan.pl):** Learn about the GHOST community, its sections (including GHOST x
  IRIM), and find contact links (e.g., Facebook, LinkedIn).
- **[IRIM Website](http://www.cie.put.poznan.pl/index-en.html):** Explore IRIM's work and find contact information for
  project supervisors.
- **GitHub Issues:** Report bugs, request features, install issues, RFCs, thoughts, etc.

For participant contact details, see the [Team](#the-team) section.

<p style="text-align: right;">(<a href="#readme-top">back to top</a>)</p>

## The Team

We are a passionate group of students from Poznan University of Technology (PUT), ranging from first-year undergraduates
to final-year graduate students.

**Team Leader:**  
Kacper Dobek ([email@example.com](mailto:email@example.com))

**Team Members:**

- Adam Mazur
- Jakub Drzymała
- Jędrzej Warczyński
- Maria Szymańska
- Mateusz Konat
- Michał Redmer
- Wiktor Kamzela
- Łukasz Osiewicz

<p style="text-align: right;">(<a href="#readme-top">back to top</a>)</p>

## Acknowledgments

Special shout-out to:

- [Best-README-Template](https://github.com/othneildrew/Best-README-Template)

<p style="text-align: right;">(<a href="#readme-top">back to top</a>)</p>

## License

[Apache-2.0](LICENSE)

<p style="text-align: right;">(<a href="#readme-top">back to top</a>)</p>
