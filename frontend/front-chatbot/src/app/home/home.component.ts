import { Component } from '@angular/core';
import { ActivatedRoute, Router, RouterModule } from '@angular/router';
import { FormsModule } from '@angular/forms';
import { AuthService } from '../auth.service';
import { CommonModule } from '@angular/common';
import { HttpClientModule } from '@angular/common/http';

@Component({
  selector: 'app-home',
  standalone: true,
  imports: [FormsModule, CommonModule, HttpClientModule, RouterModule],
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.css']
})
export class HomeComponent {
  userData = {
    username: '',
    first_name: '',
    last_name: '',
    email: '',
    password: ''
  };

  credentials = {
    username: '',
    password: ''
  };
  isLoggedIn: boolean = false;

  isLoginModalOpen = false;
  isRegisterModalOpen = false;
  loginErrorMessage: string | null = null;
  registerErrorMessage: string | null = null;
  // Variable pour stocker la destination après le login
  private redirectUrl: string | null = null;

  constructor(
    private router: Router,
    private authService: AuthService,
    private route: ActivatedRoute
  ) {}

  ngOnInit(): void {
    // Ajoutez ceci pour vérifier l'état de connexion au démarrage
    this.checkLoginStatus();

    // Souscrivez aux changements d'authentification
    this.authService.authStatusChanged.subscribe(() => {
      this.checkLoginStatus();
    });

    // Votre code existant
    this.route.queryParams.subscribe(params => {
      if (params['login'] === 'true') {
        this.openLoginModal();
      }
    });
  }
  checkLoginStatus() {
    this.isLoggedIn = !!this.authService.getToken();
  }

  openLoginModal() {
    this.isLoginModalOpen = true;
    this.isRegisterModalOpen = false;
    this.loginErrorMessage = null; // Clear previous error messages
  }

  openRegisterModal() {
    this.isRegisterModalOpen = true;
    this.isLoginModalOpen = false;
    this.registerErrorMessage = null; // Clear previous error messages
  }

  closeModal() {
    this.isLoginModalOpen = false;
    this.isRegisterModalOpen = false;
  }

  goToChat() {
    if (this.authService.getToken()) {
      // Si l'utilisateur est déjà connecté, rediriger vers chat1
      this.router.navigate(['/chat']);
    } else {
      // Sinon, ouvrir le modal de login et stocker la destination
      this.redirectUrl = '/chat';
      this.openLoginModal();
    }
  }
  goToChatDm() {
    if (this.authService.getToken()) {
      // Si l'utilisateur est déjà connecté, rediriger vers chat1
      this.router.navigate(['/chat-dm']);
    } else {
      // Sinon, ouvrir le modal de login et stocker la destination
      this.redirectUrl = '/chat-dm';
      this.openLoginModal();
    }
  
  }

  goToChatRAG() {
    if (this.authService.getToken()) {
      // Si l'utilisateur est déjà connecté, rediriger vers chat1
      this.router.navigate(['/chat-rag']);
    } else {
      // Sinon, ouvrir le modal de login et stocker la destination
      this.redirectUrl = '/chat-rag';
      this.openLoginModal();
    }
  }

  goToHome() {
    this.router.navigate(['/home']);
  }

  register(userData: { username: string; first_name: string; last_name: string; email: string; password: string }) {
    this.authService.register(userData).subscribe(
      (response) => {
        console.log('Registration successful:', response);
        this.closeModal();
        this.openLoginModal();
        this.router.navigate(['/home']);
      },
      (error) => {
        console.error('Registration error:', error);
        this.registerErrorMessage = error?.error?.message || 'Registration failed. Please try again.';
      }
    );
  }

  login(credentials: { username: string, password: string }) {
    this.authService.login(credentials).subscribe(
      (response: any) => {
        console.log('Login successful:', response);
        this.authService.saveToken(response.token);
        this.isLoggedIn = true; // Mettre à jour l'état de connexion
        this.closeModal();

        if (this.redirectUrl) {
          this.router.navigate([this.redirectUrl]);
          this.redirectUrl = null;
        } else {
          this.router.navigate(['/home']);
        }
      },
      (error) => {
        console.error('Login error:', error);
        this.loginErrorMessage = error?.error?.message || 'Login failed. Please check your credentials and try again.';
      }
    );
  }


  logout() {
    localStorage.removeItem('authToken');
    this.isLoggedIn = false;
    this.router.navigate(['/home']);
  }



  get() {
    this.authService.getUsers().subscribe((response: any) => {
      console.log('Fetched users:', response);
    });
  }
}
