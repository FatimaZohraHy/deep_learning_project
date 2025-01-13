import { Routes } from '@angular/router';
import { HomeComponent } from './home/home.component';
import { ChatComponent } from './chat/chat.component';
import { ChatDMComponent } from './chat-dm/chat-dm.component';
import { authGuard } from './auth.guard';
import { ChatRAGComponent } from './chat-rag/chat-rag.component';

export const routes: Routes = [
    {path : "chat", component : ChatComponent, canActivate: [authGuard]},
    {path : "chat-dm", component : ChatDMComponent, canActivate: [authGuard]},
    {path : "chat-rag", component : ChatRAGComponent, canActivate: [authGuard]},
    {path : "home", component : HomeComponent},
    {path : "", redirectTo : "/home", pathMatch : 'full',},
];
