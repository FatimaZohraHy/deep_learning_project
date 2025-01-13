
import { Component, inject } from '@angular/core';
import { Router } from '@angular/router';
import { FormsModule } from '@angular/forms';
import { GeminiService } from '../gemini.service';
import { AuthService } from '../auth.service';
import { CommonModule } from '@angular/common';
import { catchError, throwError, timeout, TimeoutError } from 'rxjs';
@Component({
  selector: 'app-chat-dm',
  standalone: true,
  imports: [FormsModule, CommonModule],
  templateUrl: './chat-dm.component.html',
  styleUrl: './chat-dm.component.css'
})

export class ChatDMComponent {
  title = 'front-chatbot';
  prompt: string= '';

  loading: boolean =false;
  chatHistory : any[]=[]

  authService: AuthService= inject(AuthService)
  isProcessing = false;
  
  constructor(private router: Router){
  }
    
  async SendData() {
    if (this.prompt && !this.isProcessing) {
      this.isProcessing = true;
      this.loading = true;
      const userMessage = this.prompt;
      this.prompt = '';

      this.chatHistory.push({ 
        from: 'user', 
        message: userMessage 
      });

      this.authService.threat_detection(userMessage)
        .pipe(
          timeout(300000),
          catchError(error => {
            if (error instanceof TimeoutError) {
              return throwError(() => ({
                error: 'timeout',
                message: 'Request timed out'
              }));
            }
            return throwError(() => error);
          })
        )
        .subscribe({
          next: (response) => {
            // Check if response has the expected structure
            const message = response?.response || response?.generated_text || 'No response received';
            this.chatHistory.push({ 
              from: 'bot', 
              message: message
            });
          },
          error: (error) => {
            console.error('Error:', error);
            let errorMessage = 'An error occurred. Please try again.';
            
            if (error.error === 'timeout') {
              errorMessage = 'Request timed out. Please try again with a shorter message.';
            }
            
            this.chatHistory.push({ 
              from: 'bot', 
              message: errorMessage
            });
          },
          complete: () => {
            this.loading = false;
            this.isProcessing = false;
          }
        });
    }
  }

  formatText(text: string) {
    if (!text) {
      return ''; // ou retournez une valeur par défaut appropriée
    }
    const result = text.replaceAll('*', '');
    return result;
  }

  
    // Navigate to home
    goToHome() {
      this.router.navigate(['/']);
    }
  
    // Navigate to chat (optional, since you're already on chat page)
    goToChat() {
      this.router.navigate(['/chat']);
    }
    goToChatDm() {
      this.router.navigate(['/chat-dm']);
    }
    goToChatRAG() {
      this.router.navigate(['/chat-rag']);
    }
  
    // Function to handle conversation model choice
    chooseModel(model: string) {
      console.log('Chosen model:', model);
      // Logic to switch chatbot model based on user's choice
      // For example, you can use a service to update the chatbot's behavior
    }
    
    logout() {
      // Clear the token from localStorage
      localStorage.removeItem('authToken');
  
      // Optionally, clear any other data related to the user
      // localStorage.removeItem('userData');
  
      // Redirect to the login page
      this.router.navigate(['/home']);
    }
}
