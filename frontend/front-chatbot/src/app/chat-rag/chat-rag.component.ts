import { CommonModule } from '@angular/common';
import { Component, inject } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { AuthService } from '../auth.service';
import { Router } from '@angular/router';

@Component({
  selector: 'app-chat-rag',
  standalone: true,
  imports: [FormsModule, CommonModule],
  templateUrl: './chat-rag.component.html',
  styleUrl: './chat-rag.component.css'
})
export class ChatRAGComponent {
  title = 'front-chatbot';
  prompt: string= '';

  loading: boolean =false;
  chatHistory : any[]=[]

  authService: AuthService= inject(AuthService)
  constructor(private router: Router){
  }
  async SendData() {
    if (this.prompt) {
      this.loading = true;
      const userMessage = this.prompt;
      this.prompt = '';

      // Ajouter le message de l'utilisateur à l'historique du chat
      this.chatHistory.push({ from: 'user', message: userMessage });

      try {
        // Appeler la méthode generate_response du service AuthService
        const response = await this.authService.query_rag(userMessage).toPromise();
        
        // Ajouter la réponse du bot à l'historique du chat
        this.chatHistory.push({ from: 'bot', message: response.response });
      } catch (error) {
        console.error('Error generating response:', error);
        this.chatHistory.push({ from: 'bot', message: 'Sorry, something went wrong. Please try again.' });
      } finally {
        this.loading = false;
      }
    }
  }
  formatText(text:string){
    const result = text.replaceAll('*','')
    return result
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
