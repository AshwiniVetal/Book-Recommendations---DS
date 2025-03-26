# Book-Recommendations---DS
This repository contains datasets for a Book Recommendation System, including book details, user information, and book ratings. These datasets can be used to build collaborative filtering, content-based, and hybrid recommendation models using machine learning.

## 📌 Overview
This repository contains datasets for building a **Book Recommendation System** using machine learning. The dataset includes:
- **Book Information** (title, author, publication year, publisher, and image URLs)
- **User Information** (user ID, location, and age)
- **Book Ratings** (user-book interactions)

These datasets can be used for **collaborative filtering, content-based filtering, and hybrid recommendation models**.

---

## 📊 Dataset Files & Description

| File Name      | Description |
|---------------|------------|
| `books.csv`   | Contains book details such as ISBN, title, author, publication year, publisher, and book cover image URLs |
| `users.csv`   | Contains user information such as User-ID, location, and age |
| `ratings.csv` | Contains book ratings given by users |

---

### 🔹 **Column Details**
#### 📂 `books.csv`
| Column Name             | Description |
|-------------------------|------------|
| `ISBN`                 | Unique identifier for each book |
| `Book-Title`           | Title of the book |
| `Book-Author`          | Author of the book |
| `Year-Of-Publication`  | Year the book was published |
| `Publisher`            | Publisher of the book |
| `Image-URL-S`          | URL for the small book cover image |
| `Image-URL-M`          | URL for the medium book cover image |
| `Image-URL-L`          | URL for the large book cover image |

#### 📂 `users.csv`
| Column Name | Description |
|------------|------------|
| `User-ID`  | Unique identifier for each user |
| `Location` | Location of the user (City, State, Country) |
| `Age`      | Age of the user (if available) |

#### 📂 `ratings.csv`
| Column Name | Description |
|------------|------------|
| `User-ID`  | Unique identifier for each user |
| `ISBN`     | ISBN of the rated book |
| `Book-Rating` | Rating given by the user (0-10) |

---
